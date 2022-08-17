#%%
import matplotlib
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


with open('saved_models/model11_accuracy_86.07/compressed_models.pcl', 'rb') as f:
    m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier = pickle.load(f)

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

fc1 = m1.predict(images_array2)
fc2 = m2.predict(images_array2)
pooling = pooling.predict(images_array2)

fc1 = fc1.T
fc2 = fc2.T
pooling = pooling.T


testX = np.matmul(Ax1,fc1)
testY = np.matmul(Ay1, fc2)
test_vector1 = np.concatenate((testX,testY))

testX = np.matmul(Ax2,fc1)
testY = np.matmul(Ay2, pooling)
test_vector2 = np.concatenate((testX,testY))

testX = np.matmul(Ax3,test_vector1)
testY = np.matmul(Ay3, test_vector2)
test_vector3 = np.concatenate((testX,testY))

test_vector = test_vector3.T


predicted = classifier.predict(test_vector)

# freeing memory
tf.keras.backend.clear_session()
del Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, test_vector1, test_vector2, test_vector3, classifier, m1, m2, pooling


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(labels, predicted))


#%%
import math
matching_images = []

# make sure to replace images_array in the for loop with images_array[0:10] or any slice you want if you are using an interactive 
#shell to avoid overloading your system!!!!!!!!!!!
#you can skip image by pressing 'q' or exit by pressing 'r'

for i,image in enumerate(images_array):
    label = labels[i]
    prediction = predicted[i]
    matching_images_indicies = [i for i, x in enumerate(labels) if x == prediction]
    matching_images = [images_array[i] for i in matching_images_indicies]
    nb_matches = len(matching_images)
    
    
    columns = 4
    rows = nb_matches/float(columns)
    rows = math.ceil(rows)
    
    
    ratio = np.ones(rows+2,dtype='float32')
    ratio[1] = 0.00001


    fig = plt.figure(1,tight_layout=True,figsize=(20,20))
    gs = gridspec.GridSpec(rows+2, columns,height_ratios=ratio)
    ax = fig.add_subplot(gs[0, :])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('INPUT IMAGE: True Class:{}'.format(label),fontsize=20)
    ax.imshow(image)

    ax = fig.add_subplot(gs[1, :])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("MATCHING IMAGES: Predicted Class:{}".format(prediction),fontsize=20)


    
    count = 0
    
    for row in range(2,rows+2):
        for col in range(columns):
            ax = fig.add_subplot(gs[row, col])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(matching_images[count])
            count += 1
            if count == nb_matches:
                break

    fig.align_labels()
    fig.canvas.mpl_connect('key_press_event', handle)
    mng = plt.get_current_fig_manager()

    # comment the next line if you are using ipyKernel/jupyter notebook!(Interactive shell)
    #mng.window.state('zoomed')
    plt.show()




# %%
