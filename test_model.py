from tensorflow import keras
import numpy as np
import pickle
import os
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input


m1 = keras.models.load_model('saved_models\model1/fc1_model')
m2 = keras.models.load_model('saved_models\model1/fc2_model')

Ax = np.load('saved_models\model1\Atransform.npy')
Ay = np.load('saved_models\model1\Ytransform.npy')


with open('saved_models\model1\KNN_model', 'rb') as f:

    classifier = pickle.load(f)

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

images_array = preprocess_input(images_array)

fc1 = m1.predict(images_array)
fc2 = m2.predict(images_array)

fc1 = fc1.T
fc2 = fc2.T

testX = np.matmul(Ax,fc1)
testY = np.matmul(Ay, fc2)
test_vector = np.concatenate((testX,testY))

test_vector = test_vector.T

predicted = classifier.predict(test_vector)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(labels, predicted))


