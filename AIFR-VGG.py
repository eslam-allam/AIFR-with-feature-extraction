#%%
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import keras
import graphviz
from keras.utils.vis_utils import plot_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from dcaFuse import dcaFuse
from keras import backend as K

import numpy as np
import os
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

image_directory = './datasets/FGNET/newImages/'
image_list = os.listdir(image_directory)

images_array = np.ndarray((1002,224,224,3),dtype='int32')
labels = np.arange(1002)

#fill image and label arrays
for i,image in enumerate(image_list):
    temp_image = cv2.imread(image_directory+image)
    temp_image = cv2.cvtColor(temp_image,cv2.COLOR_BGR2RGB)
    images_array[i] = temp_image
    label = int(image[0:3])-1
    labels[i] = label

# display some of the images
'''fig, ax = plt.subplots(nrows=10, ncols=10,sharex=True,sharey=True,figsize=(15,15))
i = 0
for row in ax:
    for col in row:
        col.imshow(images_array[i])
        col.title.set_text(str(labels[i]))
        i += 1
        

plt.show()'''




# split the data into train and test
X_train, X_test, y_train1, y_test1 = train_test_split(images_array, labels, test_size=0.20, random_state=33)


y_train = tf.keras.utils.to_categorical(
    y_train1, num_classes=82, dtype='float32'
)

y_test = tf.keras.utils.to_categorical(
    y_test1, num_classes=82, dtype='float32'
)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

print(X_train.shape)
print(y_train.shape)





#%%

base_model = VGG16(weights="imagenet", include_top=True, input_shape=X_train[0].shape)
base_model.trainable = False ## Not trainable weights

model = Sequential()
for layer in base_model.layers[:-1]: # go through until last layer
    model.add(layer)
model.add(Dense(82, activation='softmax'))

model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

model.fit(X_train, y_train, epochs=50,batch_size=56, callbacks=[es],validation_data=(X_test,y_test))



#%%
m1 = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
fc1 = m1.predict(X_train)

m1 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
fc2 = m1.predict(X_train)

m1 = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
flatten = m1.predict(X_train)

fc1 = fc1.T
fc2 = fc2.T
flatten = flatten.T
#%%
print("vector 1 shape :", flatten.shape)
print("vector 2 shape :", fc1.shape)
print("vector 3 shape :", fc2.shape)
print("Labels shape: ", y_train1.shape)
#%%
Xs, Ys, Ax, Ay= dcaFuse(fc1,flatten,y_train1)
fused_vector1 = np.add(Xs, Ys)

Xs, Ys, Ax, Ay= dcaFuse(fc1,fc2,y_train1)
fused_vector2 = np.add(Xs, Ys)

Xs, Ys, Ax, Ay = dcaFuse(fused_vector1, fused_vector2, y_train1)
fused_vector3 = np.add(Xs, Ys)


# %%
