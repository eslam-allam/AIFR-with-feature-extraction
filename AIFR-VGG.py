import argparse

parser = argparse.ArgumentParser(description='Train and save a DNN-MDCA model using a dataset from directory. The input images must have a shape of (224, 224, 3) and prefereably preprocessed adequatily.')
parser.add_argument('-l','--loop', required=False, action='store_true', help='loop program until desired accuracy is reached')
parser.add_argument('-es','--early-stop', required=False, action='store_true', help='stop the training early if the accuracy is not improving')
parser.add_argument('-ne','--no-excel', required=False, action='store_true', help="don't save stats to excel files")
parser.add_argument('-v','--variable-dropout', required=False, type=float, action='store', help="increase dropout after every iteration")
parser.add_argument('-d','--dropout', required=False, type=float, action='store', help="Set dropout value")
parser.add_argument('-s','--model-summary', required=False, action='store_true', help="Print summary of built TF model")
parser.add_argument('-vk','--variable-knn', required=False, action='store_true', help="keep changing number of KNN neighbors until target accuracy is reached.")
parser.add_argument('-bc','--bot-config', metavar='\b', required=False, help="Modify bot configuration file location.")
parser.add_argument('-nt','--notify-telegram', required=False, action='store_true', help="send telegram notification when training is finished")
parser.add_argument('-kn','--knn-neighbors', metavar='\b', required=False, type=int, help='number of KNN neighbors')
parser.add_argument('-sd','--save-directory', metavar='\b', required=False, type=str, help='Directory to save models')
parser.add_argument('-at','--accuracy-threshold', metavar='\b', required=False, type=float, help='Min accuracy to stop the loop')
args = parser.parse_args()

import tensorflow as tf
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras_vggface import utils
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from dcaFuse import dcaFuse
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle 
import logging
import requests
import sys
import re
from tqdm import tqdm as meter
import gc
import random

tf.get_logger().setLevel(logging.CRITICAL)
mylogs = logging.getLogger(__name__)
mylogs.setLevel(logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
streamformat = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
stream.setFormatter(streamformat)

file = logging.FileHandler("program_logs.log",encoding='utf-8')
file.setLevel(logging.INFO)
fileformat = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
file.setFormatter(fileformat)

mylogs.addHandler(stream)
mylogs.addHandler(file)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        mylogs.info('KEYBOARD INTERRUPT, PROGRAM TERMINATED')
    else:
        mylogs.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        if args.notify_telegram: notify_telegram(config_from_file=True, message=f'Model training experienced a critical error and was terninated.\nError info:\nException type: {exc_type}\nException value: {exc_value}\nTraceback: {exc_traceback}')
    sys.exit(0)

    

sys.excepthook = handle_exception
sys.setrecursionlimit(10000)


DATASET_DIRECTORY = "./datasets/FGNET/newImages/"
IMAGE_SHAPE = (224, 224, 3)
MODEL_SAVE_DIRECTORY = './saved_models/'
SAVE_MODEL_ACCURACY_THRESHOLD = 0.86
BOT_CONFIG_PATH = './bot_config.txt'
EPOCHS = 150
DROPOUT = 0.2


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
    if epoch > 50:
        lrate = 0.00001
    return lrate

def load_dataset(directory=DATASET_DIRECTORY, image_shape=IMAGE_SHAPE):
    image_list = os.listdir(directory)

    number_of_images = len(image_list)
    data_shape = (number_of_images,) + image_shape
    mylogs.debug(f'NUMBER OF IMAGES: {len(image_list)}')

    images_array = np.ndarray(
        data_shape, dtype="int32"
    )  
    labels = np.empty(number_of_images, dtype=int) 
    ages = np.empty(number_of_images, dtype=int)
    # fill image and label arrays

    mylogs.info('LOADING DATASET')
    for i, image in enumerate(meter(image_list)):
        temp_image = cv2.imread(directory + image)
        images_array[i] = temp_image
        label = image.split('.')[0].split('A')
        label, age = int(label[0]) -1 , label[1]
        ages[i] = re.sub(r'[aA-zZ]+', '', age )
        labels[i] = label
    
    num_classes = len(np.unique(labels))
    mylogs.debug(f'NUMBER OF CLASSES: {num_classes}')
    
    seed = random.randint(0, 2**32 - 1)

    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        images_array, labels, test_size=0.20, random_state=seed
    )
    y_train_ages, y_test_ages = train_test_split(ages, test_size=0.20, random_state=seed)
    

    y_test = np.array(y_test)
    labels = np.array(labels)
    
    y_train_categorical = tf.keras.utils.to_categorical(
        y_train, num_classes=num_classes, dtype="float32"
    )
    
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train = utils.preprocess_input(x_train, version=2)
    x_test = utils.preprocess_input(x_test, version=2)
    
    return x_train, y_train, x_test, y_test, y_train_categorical, y_test_ages, num_classes

def build_model(x_train, y_train,epochs=EPOCHS, early_stop=True, variable_lr=True, batch_size=128, model_summary=False, num_classes=82, drop_out= DROPOUT):

    mylogs.info('CREATING MODEL')

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
    drop1 = Dropout(drop_out)(dense1)
    dense2 = Dense(
        4096,
        activation="relu",
        name="fc2",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    )(drop1)
    drop = Dropout(drop_out)(dense2)
    predictions = Dense(num_classes, activation="softmax")(drop)  
    model = Model(inputs=base_model.input, outputs=predictions)

    
    if model_summary: 
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_lines = '\n'.join(summary_lines)
        mylogs.info(f'MODEL SUMMARY:\n{summary_lines}')

    
    mylogs.info('COMPILING MODEL')
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


    mylogs.info('TRAINING MODEL')
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.2,
        use_multiprocessing=True,
    )

    return model, history

def three_layer_MDCA(x_train, x_test,y_train, model, layer1='fc1', layer2='fc2', layer3='flatten'):

    mylogs.info('EXTRACTING FC1 VECTOR')
    m1 = Model(inputs=model.input, outputs=model.get_layer(layer1).output)
    fc1_train = m1.predict(x_train)
    fc1_test = m1.predict(x_test)

    mylogs.info('EXTRACTING FC2 VECTOR')
    m2 = Model(inputs=model.input, outputs=model.get_layer(layer2).output)
    fc2_train = m2.predict(x_train)
    fc2_test = m2.predict(x_test)

    mylogs.info('EXTRACTING FLATTEN VECTOR')
    flatten = Model(inputs=model.input, outputs=model.get_layer(layer3).output)
    flatten_train = flatten.predict(x_train)
    flatten_test = flatten.predict(x_test)

    fc1_train = fc1_train.T
    fc2_train = fc2_train.T

    fc1_test = fc1_test.T
    fc2_test = fc2_test.T

    flatten_train = flatten_train.T
    flatten_test = flatten_test.T

    mylogs.info('STAGE 1 FUSION')
    Xs, Ys, Ax1, Ay1 = dcaFuse(fc1_train, fc2_train, y_train)
    fused_vector1 = np.concatenate((Xs, Ys))

    
    testX = np.matmul(Ax1, fc1_test)
    testY = np.matmul(Ay1, fc2_test)
    test_vector1 = np.concatenate((testX, testY))

    mylogs.info('STAGE 2 FUSION')
    Xs, Ys, Ax2, Ay2 = dcaFuse(fc1_train, flatten_train, y_train)
    fused_vector2 = np.concatenate((Xs, Ys))

    testX = np.matmul(Ax2, fc1_test)
    testY = np.matmul(Ay2, flatten_test)
    test_vector2 = np.concatenate((testX, testY))

    mylogs.info('STAGE 3 FUSION')
    Xs, Ys, Ax3, Ay3 = dcaFuse(fused_vector1, fused_vector2, y_train)
    fused_vector3 = np.concatenate((Xs, Ys))

    testX = np.matmul(Ax3, test_vector1)
    testY = np.matmul(Ay3, test_vector2)
    test_vector3 = np.concatenate((testX, testY))


    fused_vector = fused_vector3.T
    test_vector = test_vector3.T

    return fused_vector, test_vector, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3

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

    writer = pd.ExcelWriter(output_directory+'/Model_accuracy_stats.xlsx', engine='xlsxwriter')
    age_based_tally.to_excel(writer, sheet_name='Age_based_tally')
    accuracy_df.to_excel(writer, sheet_name='Model_Accuracy')
    loss_df.to_excel(writer, sheet_name='Model_Loss')

    mylogs.info(f"EXPORTING EXCEL FILE TO {output_directory+'/Model_accuracy_stats.xlsx'}")
    writer.save()

def save_model(model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, accuracy,save_directory=MODEL_SAVE_DIRECTORY, model_name=None, save_excel_stats=False, y_test_ages=None, predicted=None, history=None, y_test=None):

    if not model_name: 
        saved_models = os.listdir(save_directory)
        model_name = f'model{len(saved_models)+1}_accuracy_{accuracy*100:.2f}'

    model_location = save_directory+model_name
    os.makedirs(model_location,exist_ok=False)
    models = [model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier]

    mylogs.info(f"EXPORTING MODELS TO {model_location+'/compressed_models.pcl'}")
    with open(model_location+'/compressed_models.pcl', "wb") as f:
        pickle.dump(models, f)

    if save_excel_stats:
        assert y_test_ages is not None and predicted is not None and history is not None and y_test is not None, 'y_test_ages, predicted, history, y_test are required to generate model stats'
        model_stats_to_excel(y_test_ages, predicted, history, y_test, model_location)
    
    if model_name: return model_name
    else: return 'UnnamedModel'
    
def  notify_telegram(model_name=None, accuracy=None, telegram_bot_token=None, telegram_chatID=None, bot_config_file=BOT_CONFIG_PATH, init=False, config_from_file=False, message=''):

    if init or config_from_file:
        if os.path.exists(bot_config_file):
            try:
                with open(bot_config_file,'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except FileNotFoundError:
                mylogs.error('COULD NOT LOCATE SETUP FILE')
                sys.exit(0)


            telegram_bot_token = lines[0][lines[0].find('=')+1:].strip()
            telegram_chatID = lines[1][lines[1].find('=')+1:].strip()
        else:
            mylogs.warning('SETUP FILE DOES NOT EXIST! CREATING SETUP FILE')
            if not telegram_bot_token or not telegram_chatID:
                mylogs.warning('SOME CORE VALUES NOT PROVIDED. ATTEMPTING MANUAL RETRIEVAL')
            telegram_bot_token = input('ENTER YOUR TELEGRAM BOT TOKEN: ')
            telegram_chatID = input('ENTER YOUR TELEGRAM CHAT ID: ')

            setup_text = 'telegram_bot_token = {}\ntelegram_chatID = {}'.format(telegram_bot_token, telegram_chatID)
            mylogs.info('CREATING SETUP FILE')
            with open(bot_config_file,'w', encoding='utf-8') as f:
                    f.write(setup_text)
            
            mylogs.info('SETUP FILE CREATED SUCCESSFULLY') 
            
    if init: return  telegram_bot_token, telegram_chatID

    
    if not message: 
        assert model_name and accuracy, f"INVALID VALUES FOR MODEL_NAME, ACCURACY '{model_name}', '{accuracy}'"
        message = f'Your model "{model_name}" has finished training with an accracy of: {accuracy*100:.2f}%'
    response = requests.get('https://api.telegram.org/bot{}/sendMessage'.format(telegram_bot_token),params={'text':message ,'chat_id': '{}'.format(telegram_chatID)})
    status = response.json()
    if status['ok']:
        mylogs.info('**********************STATUS:OK**********************')
        sender = status['result']['from']
        chat = status['result']['chat']
        mylogs.info('MESSAGE SENT THROUGH {} TO {} {}'.format(sender['first_name'],chat['first_name'], chat['last_name']))
        if model_name: mylogs.info('MESSAGE: {}'.format(f'Your model "{model_name}" has finished training with an accracy of: {accuracy*100:.2f}%'))
    else:
        mylogs.warning('**********************STATUS:NO OK**********************')
        mylogs.warning('MESSAGE NOT SENT!! PLEASE CHECK BOT PARAMETERS OR CHAT ID')

def main(loop=False ,early_stop=False, save_excel_stats=True,KNN_neighbors=5, save_directory=MODEL_SAVE_DIRECTORY, accuracy_threshold=SAVE_MODEL_ACCURACY_THRESHOLD, model_summary=False, drop_out=DROPOUT, variable_dropout=None, variable_knn=False):
    x_train, y_train, x_test, y_test, y_train_categorical, y_test_ages, num_classes = load_dataset()

    while True:
        model, history = build_model(x_train, y_train_categorical, early_stop=early_stop, model_summary=model_summary, num_classes=num_classes, drop_out=drop_out)
        fused_vector, test_vector, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3 = three_layer_MDCA(x_train, x_test,y_train, model)
        
        for i in range(KNN_neighbors,KNN_neighbors+10):
            mylogs.info(f'NUMBER OF kNN NEIGHBORS = {KNN_neighbors}')
            classifier = KNeighborsClassifier(n_neighbors=i)
            classifier.fit(fused_vector, y_train)

            # predict and display using DNN and KNN classifiers
            predicted = np.argmax(model.predict(x_test), axis=-1)
            mylogs.info("DNN Accuracy: {}".format(metrics.accuracy_score(y_test, predicted)))

            predicted = classifier.predict(test_vector)
            mylogs.info("DCA Accuracy: {}".format(metrics.accuracy_score(y_test, predicted)))

            DCA_accuracy = metrics.accuracy_score(y_test, predicted)

            if DCA_accuracy >= accuracy_threshold:
                model_name = save_model(model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy,
                save_excel_stats=save_excel_stats, y_test_ages=y_test_ages, predicted=predicted, history=history, y_test=y_test, save_directory=save_directory)
                return model_name, DCA_accuracy
            
            else: model_name = 'Unnamed_model'
            if not variable_knn: break

        tf.keras.backend.set_learning_phase(1)
        tf.keras.backend.clear_session()
        del model, history, fused_vector, test_vector, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3
        classifier, predicted
        if not loop: break
        elif variable_dropout is not None: 
            drop_out += variable_dropout
            mylogs.info(f'VARIABLE DROPOUT ENABLED, CURRENT DROPOUT = {drop_out}')
            assert drop_out <= 0.6, 'DROP OUT TOO HIGH, ABORTING'
        
        gc.collect(0)
        gc.collect() 
    return model_name, DCA_accuracy


        
loop, early_stop, save_excel_stats, KNN_neighbors, save_directory, accuracy_threshold, model_summary, variable_dropout, drop_out, variable_knn = False ,False, True, 5, MODEL_SAVE_DIRECTORY, SAVE_MODEL_ACCURACY_THRESHOLD, False, None, DROPOUT, False
if args.early_stop:
    early_stop = True
if args.no_excel:
    save_excel_stats = False
if args.knn_neighbors:
    KNN_neighbors = args.knn_neighbors
if args.save_directory:
    save_directory = args.save_directory
if args.model_summary:
    model_summary = True
if args.loop:
    loop = True
if args.variable_dropout:
    variable_dropout = args.variable_dropout
if args.variable_knn:
    variable_knn = True
if args.dropout:
    drop_out = args.dropout
if args.accuracy_threshold:
    accuracy_threshold = args.accuracy_threshold
    
mylogs.info(f'''STARTING TRAINING WITH THE FOLLOWING CONFIGURATION:
                             LOOP = {loop}, EARLY_STOP = {early_stop}, SAVE_EXCEL_STATS = {save_excel_stats}, MODEL_SUMMARY = {model_summary}
                             NUMBER_OF_KNN_NEIGHBOURS = {KNN_neighbors}
                             ACCURACY_THRESHOLD = {accuracy_threshold}
                             DROPOUT = {drop_out}
                             VARIABLE_DROPOUT = {variable_dropout}
                             KNN_NEIGHBORS = {KNN_neighbors}
                             VARIABLE_KNN_NEIGHBORS = {variable_knn}
                             SAVE_DIRECTORY = {save_directory}
------------------------------------------------------------------------------------------------------------------------------------------------''')

if args.notify_telegram: 
    mylogs.info('TELEGRAM NOTIFICATION ENABLED BY USER. STARTING CONFIG.')
    telegram_bot_token, telegram_chatID = notify_telegram(init=True)

model_name, accuracy = main(loop, early_stop, save_excel_stats, KNN_neighbors, save_directory, accuracy_threshold, model_summary=model_summary, variable_dropout=variable_dropout, drop_out=drop_out, variable_knn=variable_knn)

if args.notify_telegram: notify_telegram(model_name, accuracy, telegram_bot_token=telegram_bot_token, telegram_chatID=telegram_chatID)

    



