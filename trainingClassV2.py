from re import T
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


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from PIL import Image
import math
import matplotlib.gridspec as gridspec
import sys
from addImgClass import addImgClass
from sklearn.utils import shuffle

currDirectory = os.getcwd()


class trainingClassV2:
    trained = False

    def trainModel(self):
        global classifier, Ax1, Ay1, Ax2, Ay2, Ax3, Ay3, m1, m2, pooling, X_test, y_test1, predicted, dispTemp

        image_directory = "./datasets/FGNET/newImages/"
        image_list = os.listdir(image_directory)

        X_train, X_test, y_train1, y_test1, numClasses = self.imgSplit()
        # images_array = np.ndarray((numberOfimgs, 224, 224, 3), dtype="int32")
        # labels = np.arange(numberOfimgs)
        dispTemp = X_test
        y_train = tf.keras.utils.to_categorical(
            y_train1, num_classes=numClasses, dtype="float32"
        )

        y_test = tf.keras.utils.to_categorical(
            y_test1, num_classes=numClasses, dtype="float32"
        )

        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        X_train = utils.preprocess_input(X_train, version=2)
        X_test = utils.preprocess_input(X_test, version=2)

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
        predictions = Dense(numClasses, activation="softmax")(drop)
        model = Model(inputs=base_model.input, outputs=predictions)

        # freeing memory
        del base_model

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

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

        ####### dca
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

        ###########################################################################################

        Xs, Ys, Ax1, Ay1 = dcaFuse(fc1_train, fc2_train, y_train1)
        fused_vector1 = np.concatenate((Xs, Ys))

        testX = np.matmul(Ax1, fc1_test)
        testY = np.matmul(Ay1, fc2_test)
        test_vector1 = np.concatenate((testX, testY))

        ###########################################################################################

        Xs, Ys, Ax2, Ay2 = dcaFuse(fc1_train, pooling_train, y_train1)
        fused_vector2 = np.concatenate((Xs, Ys))

        testX = np.matmul(Ax2, fc1_test)
        testY = np.matmul(Ay2, pooling_test)
        test_vector2 = np.concatenate((testX, testY))

        ###########################################################################################

        Xs, Ys, Ax3, Ay3 = dcaFuse(fused_vector2, fused_vector2, y_train1)
        fused_vector3 = np.concatenate((Xs, Ys))

        ###########################################################################################

        testX = np.matmul(Ax3, test_vector1)
        testY = np.matmul(Ay3, test_vector2)
        test_vector3 = np.concatenate((testX, testY))

        ###########################################################################################

        # freeing memory
        del (
            fused_vector1,
            fused_vector2,
            test_vector1,
            test_vector2,
            fc1_train,
        )
        fc2_train, pooling_train, fc1_test, fc2_test, pooling_test  # what is the use of this line

        # clasfier phase
        fused_vector = fused_vector3.T
        test_vector = test_vector3.T

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(fused_vector, y_train1)
        predicted = classifier.predict(test_vector)

        # Model Accuracy, how often is the classifier correct?
        dcaAcc = metrics.accuracy_score(y_test1, predicted)
        print("DCA Accuracy:", dcaAcc)

        print(y_test1)
        print(predicted)

        predicted = np.argmax(model.predict(X_test), axis=-1)
        dnnAcc = metrics.accuracy_score(y_test1, predicted)
        print("DNN Accuracy:", dnnAcc)
        self.trained = True
        return dcaAcc, dnnAcc

    # Model Saving
    def saveModel(self):
        savePath = "./saved_models/model"

        sPath = "./saved_models"
        modelList = os.listdir(sPath)
        modelNum = 1
        for data in modelList:
            currNum = data.split("model")
            if int(currNum[1]) > modelNum:
                modelNum = int(currNum[1])

        modelNum = modelNum + 1
        modelNumStr = str(modelNum)
        print(modelNumStr)
        os.mkdir(sPath + "/model" + modelNumStr)
        with open(savePath + modelNumStr + "/KNN_model", "wb") as f:
            pickle.dump(classifier, f)
        np.save(savePath + modelNumStr + "/Atransform1", Ax1)
        np.save(savePath + modelNumStr + "/Ytransform1", Ay1)
        np.save(savePath + modelNumStr + "/Atransform2", Ax2)
        np.save(savePath + modelNumStr + "/Ytransform2", Ay2)
        np.save(savePath + modelNumStr + "/Atransform3", Ax3)
        np.save(savePath + modelNumStr + "/Ytransform3", Ay3)
        m1.save(savePath + modelNumStr + "/fc1_model.h5")
        m2.save(savePath + modelNumStr + "/fc2_model.h5")
        pooling.save(savePath + modelNumStr + "/pooling_model.h5")

    def predict(self, imgPath):
        values = addImgClass(imgPath)
        newPath = "./tempPics/temp.jpg"
        values.imgAfterPP.save(newPath)

        images_array = np.ndarray((1, 224, 224, 3), dtype="int32")
        temp_image = cv2.imread(newPath)
        images_array[0] = temp_image

        temp_image2 = images_array.astype("float32")
        temp_image2 = utils.preprocess_input(temp_image2, version=2)

        fc1 = m1.predict(temp_image2)
        fc2 = m2.predict(temp_image2)
        localPooling = pooling.predict(temp_image2)

        fc1 = fc1.T
        fc2 = fc2.T
        localpooling = localPooling.T

        testX = np.matmul(Ax1, fc1)
        testY = np.matmul(Ay1, fc2)
        test_vector1 = np.concatenate((testX, testY))

        testX = np.matmul(Ax2, fc1)
        testY = np.matmul(Ay2, localpooling)
        test_vector2 = np.concatenate((testX, testY))

        testX = np.matmul(Ax3, test_vector1)
        testY = np.matmul(Ay3, test_vector2)
        test_vector3 = np.concatenate((testX, testY))

        test_vector = test_vector3.T

        """with open(model + "\KNN_model", "rb") as f:

            classifier = pickle.load(f)"""

        predicted = classifier.predict(test_vector)

        print("from trained model")
        print(predicted)
        self.showOneImgprediction(predicted, imgPath)
        return predicted

    def predictWMOdel(self, imgPath, model):

        m1 = tf.keras.models.load_model(model + "/fc1_model.h5")
        m2 = tf.keras.models.load_model(model + "/fc2_model.h5")
        pooling = tf.keras.models.load_model(model + "/pooling_model.h5")

        Ax1 = np.load(model + "\Atransform1.npy")
        Ay1 = np.load(model + "\Ytransform1.npy")

        Ax2 = np.load(model + "\Atransform2.npy")
        Ay2 = np.load(model + "\Ytransform2.npy")

        Ax3 = np.load(model + "\Atransform3.npy")
        Ay3 = np.load(model + "\Ytransform3.npy")

        values = addImgClass(imgPath)
        newPath = "./tempPics/temp.jpg"
        values.imgAfterPP.save(newPath)

        images_array = np.ndarray((1, 224, 224, 3), dtype="int32")
        temp_image = cv2.imread(newPath)
        images_array[0] = temp_image

        temp_image2 = images_array.astype("float32")
        temp_image2 = utils.preprocess_input(temp_image2, version=2)

        fc1 = m1.predict(temp_image2)
        fc2 = m2.predict(temp_image2)
        localPooling = pooling.predict(temp_image2)

        fc1 = fc1.T
        fc2 = fc2.T
        localpooling = localPooling.T

        testX = np.matmul(Ax1, fc1)
        testY = np.matmul(Ay1, fc2)
        test_vector1 = np.concatenate((testX, testY))

        testX = np.matmul(Ax2, fc1)
        testY = np.matmul(Ay2, localpooling)
        test_vector2 = np.concatenate((testX, testY))

        testX = np.matmul(Ax3, test_vector1)
        testY = np.matmul(Ay3, test_vector2)
        test_vector3 = np.concatenate((testX, testY))

        test_vector = test_vector3.T

        with open(model + "\KNN_model", "rb") as f:

            classifier = pickle.load(f)

        predicted = classifier.predict(test_vector)

        print("From model " + model)
        print(predicted)
        self.showOneImgprediction(predicted, imgPath)
        return predicted

    def imgsAsDArray(self):  # saves all images of new images in an array
        FGnet_path = "./datasets/FGNET/newImages/"
        images_path = os.listdir(FGnet_path)
        finalData = []

        for i, path in enumerate(images_path):
            names = path.split(".")[0]
            names = names.split("A")
            names[1] = names[1].replace("a", "")
            names[1] = names[1].replace("b", "")
            # finalData.append([int(names[0]), int(names[1])])
            finalData.append([names[0], [int(names[1]), path]])

        # print(finalData)

        collectingData = {}
        # collectingData["sd"] = []

        for data in finalData:
            try:
                collectingData[data[0]].append(data[1])
            except:
                # print("An exception occurred")
                collectingData[data[0]] = [data[1]]
        return collectingData

    def displayPredictions(self):
        print(type(dispTemp))
        for idx, data in enumerate(dispTemp):
            print("corr")
            print(y_test1[idx])
            print("pred")
            print(predicted[idx])

            cv2.imshow("graycsale image", np.uint8(data))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def showOneImgprediction(self, imgPrediction, ogImg):
        def handle(event):
            if event.key == "r":
                plt.close("all")

        collData = self.imgsAsDArray()
        FGnet_path = "./datasets/FGNET/newImages/"

        columns = 4
        predctedLabel = imgPrediction + 1
        predictedLabelString = str(predctedLabel[0])

        if len(predictedLabelString) == 1:
            stringLabel = "00" + predictedLabelString
        elif len(predictedLabelString) == 2:
            stringLabel = "0" + predictedLabelString
        else:
            stringLabel = predictedLabelString

        print(stringLabel)

        rows = len(collData[stringLabel]) / float(columns)
        rows = math.ceil(rows)

        ratio = np.ones(rows + 2, dtype="float32")
        ratio[1] = 0.00001

        # print(ogImg)
        # print(type(ogImg))
        temp = cv2.imread(ogImg[0])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        fig = plt.figure(1, tight_layout=True, figsize=(20, 20))
        gs = gridspec.GridSpec(rows + 2, columns, height_ratios=ratio)
        ax = fig.add_subplot(gs[0, :])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("INPUT IMAGE: ", fontsize=20)
        ax.imshow(temp)

        ax = fig.add_subplot(gs[1, :])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(
            "MATCHING IMAGES: Predicted Class:{}".format(predctedLabel), fontsize=20
        )

        count = 0
        for row in range(2, rows + 2):
            for col in range(columns):
                ax = fig.add_subplot(gs[row, col])
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                temp2 = cv2.imread(FGnet_path + collData[stringLabel][count][1])
                ax.imshow(temp2)
                count += 1
                if count == len(collData[stringLabel]):
                    break

        fig.align_labels()
        fig.canvas.mpl_connect("key_press_event", handle)
        mng = plt.get_current_fig_manager()

        # comment the next line if you are using ipyKernel/jupyter notebook!(Interactive shell)
        mng.window.state("zoomed")
        plt.show()

    breakLoop = False

    def displayAllpredections(self):
        self.breakLoop = False

        def handle(event):
            if event.key == "r":
                self.breakLoop = True
                plt.close("all")

        collData = self.imgsAsDArray()
        FGnet_path = "./datasets/FGNET/newImages/"

        for idx, img in enumerate(dispTemp):
            columns = 4
            actualLabel = y_test1[idx] + 1
            predctedLabel = predicted[idx] + 1

            actualLabelString = str(predctedLabel)
            # stingLabel
            print(actualLabel)
            print(predctedLabel)

            if len(actualLabelString) == 1:
                stringLabel = "00" + actualLabelString
            elif len(actualLabelString) == 2:
                stringLabel = "0" + actualLabelString
            else:  # new line not importnat
                stringLabel = actualLabelString
            # print(collData["0" + str(actualLabel)])
            rows = len(collData[stringLabel]) / float(columns)
            rows = math.ceil(rows)

            ratio = np.ones(rows + 2, dtype="float32")
            ratio[1] = 0.00001

            # temp = cv2.imread(FGnet_path + collData["001"][1][1])
            temp = np.uint8(img)

            fig = plt.figure(1, tight_layout=True, figsize=(20, 20))
            gs = gridspec.GridSpec(rows + 2, columns, height_ratios=ratio)
            ax = fig.add_subplot(gs[0, :])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title("INPUT IMAGE: True Class:{}".format(actualLabel), fontsize=20)
            ax.imshow(temp)

            ax = fig.add_subplot(gs[1, :])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(
                "MATCHING IMAGES: Predicted Class:{}".format(predctedLabel), fontsize=20
            )

            count = 0
            for row in range(2, rows + 2):
                for col in range(columns):
                    ax = fig.add_subplot(gs[row, col])
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    temp2 = cv2.imread(FGnet_path + collData[stringLabel][count][1])
                    ax.imshow(temp2)
                    count += 1
                    if count == len(collData[stringLabel]):
                        break

            fig.align_labels()
            fig.canvas.mpl_connect("key_press_event", handle)
            mng = plt.get_current_fig_manager()

            # comment the next line if you are using ipyKernel/jupyter notebook!(Interactive shell)
            mng.window.state("zoomed")
            plt.show()
            if self.breakLoop == True:
                break

    def getLastModel(self):
        try:
            savePath = "./saved_models/model"

            sPath = "./saved_models"
            modelList = os.listdir(sPath)
            modelNum = 0
            for data in modelList:
                currNum = data.split("model")
                if int(currNum[1]) > modelNum:
                    modelNum = int(currNum[1])

            modelNumStr = str(modelNum)
            if modelNum > 0:
                return savePath + modelNumStr
            else:
                return False

        except:
            return False

    # read pre proccessed images (newImages) from directory then perform 0.8/0.2 split
    def imgSplit(self):
        FGnet_path = "./datasets/FGNET/newImages/"
        images_path = os.listdir(FGnet_path)

        finalData = []
        totalImages = 0
        for i, path in enumerate(images_path):
            temp_image = cv2.imread(FGnet_path + path)
            totalImages += 1
            names = path.split(".")[0]
            names = names.split("A")
            names[1] = names[1].replace("a", "")
            names[1] = names[1].replace("b", "")
            finalData.append(
                [names[0], temp_image]
            )  ## just need to change here and at the start of the loop to extract the images

        # print(finalData)
        collectingData = {}
        # collectingData["sd"] = []

        for data in finalData:
            try:
                collectingData[data[0]].append(data[1])
            except:
                # print("An exception occurred")
                collectingData[data[0]] = [data[1]]
        X_train = []
        X_test = []
        y_train1 = []  # labels
        y_test1 = []  # labels

        for tags in collectingData:

            numOfImgs = len(collectingData[tags])
            testSetCount = round(numOfImgs * 0.15)
            trainSetCount = numOfImgs - testSetCount
            for i, imags in enumerate(collectingData[tags]):
                if i < trainSetCount:
                    X_train.append(imags)
                    y_train1.append(int(tags) - 1)
                else:
                    X_test.append(imags)
                    y_test1.append(int(tags) - 1)
            # print("-")
        images_array1 = np.ndarray((len(X_train), 224, 224, 3), dtype="int32")
        images_array2 = np.ndarray((len(X_test), 224, 224, 3), dtype="int32")

        for i, ins in enumerate(X_train):
            images_array1[i] = ins
        for i, ins in enumerate(X_test):
            images_array2[i] = ins

        X_train = images_array1
        X_test = images_array2
        y_train1 = np.array(y_train1)
        y_test1 = np.array(y_test1)
        """print(type(X_train))
        print(type(y_train1))
        print(type(X_test))
        print(type(y_test1))"""
        # print(len(X_train), len(y_train1))
        # print(len(X_test), len(y_test1))
        # print(totalImages)
        # print(totalImages)
        # print(len(collectingData))
        # print(len(collectingData["002"]))
        X_train, y_train1 = shuffle(X_train, y_train1, random_state=33)
        X_test, y_test1 = shuffle(X_test, y_test1, random_state=33)

        """print(type(X_train))
        print(type(y_train1))
        print(type(X_test))
        print(type(y_test1))"""
        return X_train, X_test, y_train1, y_test1, len(collectingData)
        # fill image and label arrays
        """for i, image in enumerate(image_list):
            temp_image = cv2.imread(image_directory + image)
            images_array[i] = temp_image
            label = int(image[0:3]) - 1
            labels[i] = label"""


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


train = trainingClassV2()
train.trainModel()

# train.trainModel(5, 5)
# train.displayPredictions()
train.displayAllpredections()
# temp, temp2, temp3, temp4 = train.imgSplit()
# print(temp2)
