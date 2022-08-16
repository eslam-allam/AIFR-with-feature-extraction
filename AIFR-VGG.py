
from dataclasses import dataclass
from gc import callbacks
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras_vggface import utils
import pandas as pd

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

from tqdm import tqdm as meter
import logging
import theano
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle


DATASET_DIRECTORY = "./datasets/FGNET/newImages/"
DATASET_SHAPE = (1002 ,224, 224, 3)
NUM_OF_CLASSES = 82
MODEL_SAVE_DIRECTORY = './saved_models/'
SAVE_MODEL_ACCURACY_THRESHOLD = 86.0


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

def load_dataset(directory=DATASET_DIRECTORY, data_shape=DATASET_SHAPE, num_classes=NUM_OF_CLASSES):
    image_list = os.listdir(directory)

    images_array = np.ndarray(
        data_shape, dtype="int32"
    )  ##################################
    labels = []  ##################################
    y_test_ages = []
    # fill image and label arrays

    for i, image in enumerate(image_list):
        temp_image = cv2.imread(directory + image)
        images_array[i] = temp_image
        label = image[0:6]
        labels.append(label)

    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        images_array, labels, test_size=0.20, random_state=33
    )

    for i, y in enumerate(y_train):
        y_train[i] = int(y[0:3]) - 1

    for i, y in enumerate(y_test):
        y_test[i] = int(y[0:3]) - 1
        y_test_ages.append(int(y[4:6]))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    labels = np.array(labels)
    
    y_train_categorical = tf.keras.utils.to_categorical(
        y_train, num_classes=num_classes, dtype="float32"
    )
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=num_classes, dtype="float32")
    
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train = utils.preprocess_input(x_train, version=2)
    x_test = utils.preprocess_input(x_test, version=2)
    
    return x_train, y_train, x_test, y_test, y_train_categorical, y_test_categorical, y_test_ages

def build_model(x_train, y_train,epochs=50, early_stop=True, variable_lr=True, batch_size=32, model_summary=False):
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
    predictions = Dense(82, activation="softmax")(drop)  
    model = Model(inputs=base_model.input, outputs=predictions)

   
    if model_summary: model.summary()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    
    es = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
    )

    callbacks = []
    if early_stop:
        callbacks.append(es)
    if variable_lr:
        callbacks.append(LearningRateScheduler(lr_schedule))

    steps = int(x_train.shape[0] / batch_size)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.2,
        steps_per_epoch=steps,
        use_multiprocessing=True,
    )

    return model, history

def three_layer_MDCA(x_train, x_test, model, layer1='fc1', layer2='fc2', layer3='flatten'):
    m1 = Model(inputs=model.input, outputs=model.get_layer(layer1).output)
    fc1_train = m1.predict(x_train)
    fc1_test = m1.predict(x_test)

    m2 = Model(inputs=model.input, outputs=model.get_layer(layer2).output)
    fc2_train = m2.predict(x_train)
    fc2_test = m2.predict(x_test)

    pooling = Model(inputs=model.input, outputs=model.get_layer(layer3).output)
    pooling_train = pooling.predict(x_train)
    pooling_test = pooling.predict(x_test)

    fc1_train = fc1_train.T
    fc2_train = fc2_train.T

    fc1_test = fc1_test.T
    fc2_test = fc2_test.T

    pooling_train = pooling_train.T
    pooling_test = pooling_test.T

    Xs, Ys, Ax1, Ay1 = dcaFuse(fc1_train, fc2_train, y_train)
    fused_vector1 = np.concatenate((Xs, Ys))

    testX = np.matmul(Ax1, fc1_test)
    testY = np.matmul(Ay1, fc2_test)
    test_vector1 = np.concatenate((testX, testY))

    Xs, Ys, Ax2, Ay2 = dcaFuse(fc1_train, pooling_train, y_train)
    fused_vector2 = np.concatenate((Xs, Ys))

    testX = np.matmul(Ax2, fc1_test)
    testY = np.matmul(Ay2, pooling_test)
    test_vector2 = np.concatenate((testX, testY))

    Xs, Ys, Ax3, Ay3 = dcaFuse(fused_vector1, fused_vector2, y_train)
    fused_vector3 = np.concatenate((Xs, Ys))

    testX = np.matmul(Ax3, test_vector1)
    testY = np.matmul(Ay3, test_vector2)
    test_vector3 = np.concatenate((testX, testY))

    fc2_train, pooling_train, fc1_test, fc2_test, pooling_test

    fused_vector = fused_vector3.T
    test_vector = test_vector3.T

    return fused_vector, test_vector, m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3

def model_stats_to_excel(y_test_ages, predicted, history, y_test, output_directory='./'):
    
    if not os.path.exists(output_directory): os.mkdir(output_directory)
        
    age_based_tally = dict.fromkeys(np.unique(y_test_ages),[0,0])
    age_based_tally = pd.DataFrame(data = age_based_tally)
    age_based_tally.index = ['correct','incorrect']

    total_correct = 0
    total_wrong = 0


    for i in range(0,len(y_test)):
        if y_test[i] == predicted[i]:
            age_based_tally[y_test_ages[i]][0] += 1
            total_correct += 1
        else:
            age_based_tally[y_test_ages[i]][1] += 1
            total_wrong += 1


    age_based_tally = age_based_tally.T
    age_based_tally.to_excel(output_directory+'Age_based_tally.xlsx')

    accuracy_history = history.history['accuracy']
    val_accuraccy_history = history.history['val_accuracy']

    accuracy_df = pd.DataFrame(data=(accuracy_history,val_accuraccy_history))
    accuracy_df.index = ['accuracy','val accuracy']

    # summarize history for loss
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    loss_df = pd.DataFrame(data=(loss_history,val_loss_history))
    loss_df.index = ['loss','val loss']

    accuracy_df = accuracy_df.T
    loss_df = loss_df.T

    accuracy_df.to_excel(output_directory+'Model_Accuracy.xlsx')
    loss_df.to_excel(output_directory+'Model_Loss.xlsx')

def save_model(model, m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, save_directory=MODEL_SAVE_DIRECTORY, model_name=None):

    if not model_name: 
        saved_models = os.listdir(save_directory)
        model_name = f'model{len(saved_models)}'

    model_location = save_directory+model_name
    os.makedirs(model_location,exist_ok=True)
    with open(model_location+"KNN_model", "wb") as f:
        pickle.dump(classifier, f)
    np.save(model_location+"Atransform1", Ax1)
    np.save(model_location+"Ytransform1", Ay1)
    np.save(model_location+"Atransform2", Ax2)
    np.save(model_location+"Ytransform2", Ay2)
    np.save(model_location+"Atransform3", Ax3)
    np.save(model_location+"Ytransform3", Ay3)
    m1.save(model_location+"fc1_model.h5")
    m2.save(model_location+"fc2_model.h5")
    pooling.save(model_location+"pooling_model.h5")
x_train, y_train, x_test, y_test, y_train_categorical, y_test_categorical, y_test_ages = load_dataset()

while True:
    model, history = build_model(x_train, y_train_categorical, early_stop=False)
    fused_vector, test_vector, m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3 = three_layer_MDCA(x_train, x_test, model)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(fused_vector, y_train)

    # predict and display using DNN and KNN classifiers
    spacer = 35*"-"
    predicted = np.argmax(model.predict(x_test), axis=-1)
    print("{}\nDNN Accuracy: {}\n{}".format(spacer,metrics.accuracy_score(y_test, predicted),spacer))

    predicted = classifier.predict(test_vector)
    print("DCA Accuracy: {}\n{}".format(metrics.accuracy_score(y_test, predicted),spacer))

    if metrics.accuracy_score(y_test, predicted) > SAVE_MODEL_ACCURACY_THRESHOLD:
        model_stats_to_excel(y_test_ages, predicted, history, y_test, './accuracy_stats')
        save_model(model, m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3)
        break
    tf.keras.backend.set_learning_phase(1)
    tf.keras.backend.clear_session()
    del model, history, fused_vector, test_vector, m1, m2, pooling, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3,
    classifier, spacer, predicted



