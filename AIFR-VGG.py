#%%
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import keras
import graphviz
from keras.utils.vis_utils import plot_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

from dcaFuse import dcaFuse
from keras import backend as K

import numpy as np
import os
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from tensorflow.keras import regularizers
import pickle




image_directory = './datasets/FGNET/newImages/'
image_list = os.listdir(image_directory)

images_array = np.ndarray((1002,224,224,3),dtype='int32')
labels = np.arange(1002)

#fill image and label arrays
for i,image in enumerate(image_list):
    temp_image = cv2.imread(image_directory+image)
    images_array[i] = temp_image
    label = int(image[0:3])-1
    labels[i] = label


# split the data into train and test
X_train, X_test, y_train1, y_test1 = train_test_split(images_array, labels, test_size=0.20, random_state=33)


numClasses = 82

y_train = tf.keras.utils.to_categorical(
    y_train1, num_classes=numClasses, dtype='float32'
)

y_test = tf.keras.utils.to_categorical(
    y_test1, num_classes=numClasses, dtype='float32'
)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

print(X_train.shape)
print(y_train.shape)





#%%
base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
base_model.trainable = False ## Not trainable weights


x = base_model.layers[-1].output 
dense1 = Dense(4096, activation = "relu",name='fc1',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
dense2 = Dense(4096,activation='relu',name='fc2',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(dense1)
drop = Dropout(0.2)(dense2)
predictions = Dense(82,activation='softmax')(drop)
model = Model(inputs = base_model.input, outputs = predictions)




model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
#%%
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

model.fit(X_train, y_train, epochs=50,batch_size=32, callbacks=[es],validation_data=(X_test,y_test))



#%%

m1 = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
fc1_train = m1.predict(X_train)
fc1_test = m1.predict(X_test)

m2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
fc2_train = m1.predict(X_train)
fc2_test = m1.predict(X_test)

fc1_train = fc1_train.T
fc2_train = fc2_train.T

fc1_test = fc1_test.T
fc2_test = fc2_test.T
#%%
print("vector 1 shape :", fc1_train.shape)
print("vector 2 shape :", fc2_train.shape)
print("Labels shape: ", y_train1.shape)

#%%


Xs, Ys, Ax, Ay= dcaFuse(fc1_train,fc2_train,y_train1)
fused_vector = np.concatenate((Xs, Ys))

testX = np.matmul(Ax,fc1_test)
testY = np.matmul(Ay, fc2_test)
test_vector = np.concatenate((testX,testY))

print("fusion Done")
print("fused_vector shape: ",fused_vector.shape)
print("test_vector shape: ",test_vector.shape)

#%%

fused_vector = fused_vector.T
test_vector = test_vector.T


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(fused_vector,y_train1)
predicted = classifier.predict(test_vector)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test1, predicted))

#check the dcaFuse mat file sample use cuz u fcked up dog

# %%
with open('./saved_models/model1/KNN_model', 'wb') as f:
    pickle.dump(classifier, f) 
np.save('./saved_models/model1/Atransform',Ax)                     
np.save('./saved_models/model1/Ytransform',Ay)      
m1.save('./saved_models/model1/fc1_model')
m2.save('./saved_models/model1/fc2_model')




# %%
