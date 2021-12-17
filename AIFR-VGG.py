#%%
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras_vggface import utils

import keras
import graphviz
from keras.utils.vis_utils import plot_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from dcaFuse import dcaFuse
from keras import backend as K

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

import pickle


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 5:
        lrate = 0.0005
    if epoch > 15:
        lrate = 0.0003
    if epoch > 20:
        lrate = 0.0001
    if epoch > 25:
        lrate = 0.00005
    return lrate


image_directory = "./datasets/FGNET/newImages/"
image_list = os.listdir(image_directory)

images_array = np.ndarray(
    (1002, 224, 224, 3), dtype="int32"
)  ##################################
labels = []  ##################################
y_test_ages = []
# fill image and label arrays
for i, image in enumerate(image_list):
    temp_image = cv2.imread(image_directory + image)
    images_array[i] = temp_image
    label = image[0:6]
    labels.append(label)


# split the data into train and test
X_train, X_test, y_train1, y_test1 = train_test_split(
    images_array, labels, test_size=0.20, random_state=33
)

for i, y in enumerate(y_train1):
    y_train1[i] = int(y[0:3]) - 1

for i, y in enumerate(y_test1):
    y_test1[i] = int(y[0:3]) - 1
    y_test_ages.append(int(y[4:6]))

y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)
labels = np.array(labels)
# freeing memory
del (
    labels,
    temp_image,
    images_array,
    label,
    image,
    image_directory,
    image_list,
)

numClasses = 82  ##################################

y_train = tf.keras.utils.to_categorical(
    y_train1, num_classes=numClasses, dtype="float32"
)

y_test = tf.keras.utils.to_categorical(y_test1, num_classes=numClasses, dtype="float32")
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = utils.preprocess_input(X_train, version=2)
X_test = utils.preprocess_input(X_test, version=2)

print(X_train.shape)
print(y_train.shape)


#%%
base_model = VGGFace(
    model="senet50", include_top=True, input_shape=(224, 224, 3), pooling="avg"
)
base_model.trainable = False  ## Not trainable weights


x = base_model.layers[-2].output
dense1 = Dense(
    4096,
    activation="relu",
    name="fc1",
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
)(x)
dense2 = Dense(
    4096,
    activation="relu",
    name="fc2",
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
)(dense1)
drop = Dropout(0.3)(dense2)
predictions = Dense(82, activation="softmax")(drop)  ##################################
model = Model(inputs=base_model.input, outputs=predictions)

# freeing memory
del base_model


model.summary()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
#%%
es = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
)
steps = int(X_train.shape[0] / 32)
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    callbacks=[es, LearningRateScheduler(lr_schedule)],
    validation_split=0.2,
    steps_per_epoch=steps,
    use_multiprocessing=True,
)


#%%

m1 = Model(inputs=model.input, outputs=model.get_layer("fc1").output)
fc1_train = m1.predict(X_train)
fc1_test = m1.predict(X_test)

m2 = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
fc2_train = m1.predict(X_train)
fc2_test = m1.predict(X_test)

pooling = Model(inputs=model.input, outputs=model.get_layer("flatten").output)
pooling_train = pooling.predict(X_train)
pooling_test = pooling.predict(X_test)


fc1_train = fc1_train.T
fc2_train = fc2_train.T

fc1_test = fc1_test.T
fc2_test = fc2_test.T

pooling_train = pooling_train.T
pooling_test = pooling_test.T
#%%
print("vector 1 shape :", fc1_train.shape)
print("vector 2 shape :", fc2_train.shape)
print("Labels shape: ", y_train1.shape)

#%%


Xs, Ys, Ax1, Ay1 = dcaFuse(fc1_train, fc2_train, y_train1)
fused_vector1 = np.concatenate((Xs, Ys))

testX = np.matmul(Ax1, fc1_test)
testY = np.matmul(Ay1, fc2_test)
test_vector1 = np.concatenate((testX, testY))

print("fused_vector1: ", fused_vector1.shape)
print("test_vector1: ", test_vector1.shape)


Xs, Ys, Ax2, Ay2 = dcaFuse(fc1_train, pooling_train, y_train1)
fused_vector2 = np.concatenate((Xs, Ys))

testX = np.matmul(Ax2, fc1_test)
testY = np.matmul(Ay2, pooling_test)
test_vector2 = np.concatenate((testX, testY))

print("fused_vector2: ", fused_vector2.shape)
print("test_vector2: ", test_vector2.shape)

Xs, Ys, Ax3, Ay3 = dcaFuse(fused_vector2, fused_vector2, y_train1)
fused_vector3 = np.concatenate((Xs, Ys))

print("fused_vector3: ", fused_vector3.shape)

testX = np.matmul(Ax3, test_vector1)
testY = np.matmul(Ay3, test_vector2)
test_vector3 = np.concatenate((testX, testY))

print("fusion Done")
print("fused_vector shape: ", fused_vector3.shape)
print("test_vector shape: ", test_vector3.shape)

# freeing memory
del (
    fused_vector1,
    fused_vector2,
    test_vector1,
    test_vector2,
    fc1_train,
)
fc2_train, pooling_train, fc1_test, fc2_test, pooling_test

#%%

fused_vector = fused_vector3.T
test_vector = test_vector3.T


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(fused_vector, y_train1)
predicted = classifier.predict(test_vector)


from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("DCA Accuracy:", metrics.accuracy_score(y_test1, predicted))

predicted = np.argmax(model.predict(X_test), axis=-1)
print("DNN Accuracy:", metrics.accuracy_score(y_test1, predicted))

#%%
age_dict_correct = dict.fromkeys(np.unique(y_test_ages),0)
age_dict_wrong = age_dict_correct.copy()
total = 0
for i in range(0,len(y_test1)-1):
    if y_test1[i] == predicted[i]:
        age_dict_correct[y_test_ages[i]] = age_dict_correct[y_test_ages[i]] + 1
        total +=1
    else:
        age_dict_wrong[y_test_ages[i]] = age_dict_wrong[y_test_ages[i]] + 1

# %%
with open("./saved_models/model3/KNN_model", "wb") as f:
    pickle.dump(classifier, f)
np.save("./saved_models/model3/Atransform1", Ax1)
np.save("./saved_models/model3/Ytransform1", Ay1)
np.save("./saved_models/model3/Atransform2", Ax2)
np.save("./saved_models/model3/Ytransform2", Ay2)
np.save("./saved_models/model3/Atransform3", Ax3)
np.save("./saved_models/model3/Ytransform3", Ay3)
m1.save("./saved_models/model3/fc1_model.h5")
m2.save("./saved_models/model3/fc2_model.h5")
pooling.save("./saved_models/model3/pooling_model.h5")


# %%
