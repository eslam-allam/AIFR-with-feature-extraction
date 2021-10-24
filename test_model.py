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

%matplotlib inline



tf.autograph.set_verbosity(0)

# shuffle images and labels in the exact same way
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


m1 = tf.keras.models.load_model('saved_models\model4/fc1_model.h5')
m2 = tf.keras.models.load_model('saved_models\model4/fc2_model.h5')
pooling = tf.keras.models.load_model('saved_models\model4/pooling_model.h5')


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



Ax1 = np.load('saved_models\model4\Atransform1.npy')
Ay1 = np.load('saved_models\model4\Ytransform1.npy')

Ax2 = np.load('saved_models\model4\Atransform2.npy')
Ay2 = np.load('saved_models\model4\Ytransform2.npy')

Ax3 = np.load('saved_models\model4\Atransform3.npy')
Ay3 = np.load('saved_models\model4\Ytransform3.npy')


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




with open('saved_models\model4\KNN_model', 'rb') as f:

    classifier = pickle.load(f)


predicted = classifier.predict(test_vector)

# freeing memory
tf.keras.backend.clear_session()
del Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, test_vector1, test_vector2, test_vector3, classifier, m1, m2, pooling


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(labels, predicted))


#%%
matching_images = []
for i,image in enumerate(images_array[0:10]):
    label = labels[i]
    prediction = predicted[i]
    matching_images_indicies = [i for i, x in enumerate(labels) if x == prediction]
    matching_images = [images_array[i] for i in matching_images_indicies]
    print("\n----------------\nInput Image: True Class = {}".format(label))
    

    print("Matching Images: Predicted Class = {}\n----------------".format(prediction))
    columns = 4
    rows = 1
    nb_matches = len(matching_images)

    
    while True:
        rows_enough = True

        if rows < nb_matches/columns:
            rows_enough = False
            rows +=1
        
        if rows_enough == True: break
    
    ratio = np.ones(rows+2,dtype='float32')
    ratio[1] = 0.00001


    fig = plt.figure(tight_layout=True,figsize=(40,40))
    gs = gridspec.GridSpec(rows+2, columns,height_ratios=ratio)
    ax = fig.add_subplot(gs[0, :])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('INPUT IMAGE: True Class:{}'.format(label),fontsize=80)
    ax.imshow(image)

    ax = fig.add_subplot(gs[1, :])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("MATCHING IMAGES: Predicted Class:{}".format(prediction),fontsize=80)


    
    count = 0
    
    for row in range(2,rows):
        for col in range(columns):
            ax = fig.add_subplot(gs[row, col])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(matching_images[count])
            count += 1

    fig.align_labels()
    
    
    plt.show()




# %%
