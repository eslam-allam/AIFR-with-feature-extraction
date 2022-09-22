#%%
import matplotlib
from AIFR_VGG import predict
import tensorflow as tf
import numpy as np
import pickle
import os
import cv2
from keras_vggface import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

#%matplotlib inline

def handle(event):
    if event.key == 'r':
        sys.exit(0)



tf.get_logger().setLevel('INFO')

# shuffle images and labels in the exact same way
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


with open('saved_models/model10_accuracy_87.06/compressed_models.pcl', 'rb') as f:
    model,Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier = pickle.load(f)


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

images_array, labels = unison_shuffled_copies(images_array,labels)

images_array2 = images_array.astype('float32')
images_array2 = utils.preprocess_input(images_array2,version=2)
#%%
predicted = predict(images_array2,  model,Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier)

# freeing memory


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(labels, predicted))




# %%
