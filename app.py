import ctypes
from multiprocessing import Process , shared_memory, Value, Manager
import multiprocessing

import os
from flask import Flask, Response, render_template, request
import cv2
import numpy as np
from mediapipecam import face_mesh
from allign import proccess_image
from AIFR_VGG import  MODEL_SAVE_DIRECTORY, main, save_model
import AIFR_VGG
import logging
import traceback
from ctypes import c_char_p
import tensorflow as tf
import gc
from numba import cuda



model, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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
mylogs.addHandler(logging.getLogger('AIFR_VGG'))



NEW_IMAGE_DIRECTORY = './datasets/FGNET/newImages'
HEADERS = {'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'}
video_error = cv2.imread('guiImages/video_error.png')
ret, video_error = cv2.imencode('.jpg',video_error)
video_error = video_error.tobytes()

app = Flask(__name__)

def switch_duplicates(duplicates:int):

    if duplicates == 0:
        return ''
    elif duplicates == 1:
        return 'b'
    elif duplicates == 2:
        return 'c'
    elif duplicates == 3:
        return 'd'
    elif duplicates == 4:
        return 'e'
    elif duplicates == 5:
        return 'f'
    elif duplicates == 6:
        return 'g'
    elif duplicates == 7:
        return 'h'
    elif duplicates == 8:
        return 'i'
    elif duplicates == 9:
        return 'j'
    elif duplicates == 10:
        return 'k'
    elif duplicates == 11:
        return 'l'
    elif duplicates == 12:
        return 'm'
    elif duplicates == 13:
        return 'n'
    elif duplicates == 14:
        return 'o'
    elif duplicates == 15:
        return 'p'

#camera=cv2.VideoCapture(2)

def string_to_bool(expression : str):
    if expression.casefold() == 'true':
        return True
    else: return False

def testDevice(source):
    cap = cv2.VideoCapture(source) 
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return False
    return True



current_pic = None
name = False


def generate_frames():
    tf.keras.backend.clear_session()
    gc.collect()
    device = cuda.get_current_device()
    device.reset()
    
    shm = shared_memory.SharedMemory(create=True, size=1150528, name='processed_frame_memory')
    shared_processed_frame = np.ndarray((224, 224, 3), dtype='int32', buffer=shm.buf)
    prediction = Value('i', 0)
    parallel = Process(target= AIFR_VGG.predict, args=(prediction, active_model,))
    parallel.start()
    try:

        if  testDevice(0): camera = cv2.VideoCapture(0)   
        else:
            return
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        processed_frame = None

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2
        name = ''

        while True:
            
            try:
                
            ## read the camera frame
                success,frame=camera.read()

                

                processed_frame = proccess_image(frame)
                if processed_frame is None: continue
                processed_frame = np.repeat(processed_frame[..., np.newaxis], 3, -1)
                
                
                
                if prediction.value != 0:
                    shared_processed_frame[:] = processed_frame[:]
                    
 
                if prediction.value <= 0:
                    name = 'Analyzing Face'
                else: 
                    name = str(prediction.value)
               

                frame, top_right_landmark = face_mesh(frame)
                
                
                cv2.putText(frame,name, 
                top_right_landmark, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
                
                if not success:
                    break
                else:
                    _,buffer=cv2.imencode('.jpg',frame)
                    frame=buffer.tobytes()


                
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except AttributeError as e:
                print(e)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + video_error + b'\r\n')
    except GeneratorExit:
        print('joining child')
        parallel.kill()
        print('closed')
        camera.release()
        shm.close()
        shm.unlink()
   
    
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/video')
def video():
   
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame', headers=HEADERS)
    
@app.route('/uploadimage', methods=['POST', 'OPTIONS'])
def upload_image():
    image = request.files.get('files.myImage')
    image_name = image.filename
    image = np.fromfile(image)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = proccess_image(image)

    if image is None:
        print('face not found')
        return Response('',status=404, headers=HEADERS)



    image_list = sorted(os.listdir(NEW_IMAGE_DIRECTORY))
    username, age = image_name.split('.')[0].split('A')
    usernumber = ''
    for name in image_list:
        if username in name:
            usernumber = name[0:3]
            break
    if not usernumber:
        usernumber = str(int(image_list[-1][0:3]) + 1).zfill(3)
    
    
    duplicates = 0
    for name in image_list:
        if usernumber in name and age in name:
            duplicates += 1
    print(duplicates)
    age += switch_duplicates(duplicates)
    image_name = f'{usernumber}-{username}A{age}.jpg'

    cv2.imwrite(f'{NEW_IMAGE_DIRECTORY}/{image_name}', image)

    return Response('',status=200, headers=HEADERS)

    
@app.route('/trainmodel', methods=['GET'])
def trainmodel():
    
    global model, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory, active_model
    
    args = request.args
    loop = string_to_bool(args['loop'])
    es = string_to_bool(args['es'])
    estats = string_to_bool(args['estats'])
    vknn = string_to_bool(args['vknn'])
    at = float(args['at']) / 100
    dropout = float(args['dropout'])
    knn = int(float(args['knn']))
    vdropout = float(args['vdropout'])
    save = string_to_bool(args['save'])
    auto_save = string_to_bool(args['autosave'])

    try:
        if not save:
            if not auto_save:
                model, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory = main(loop, es, estats, knn, accuracy_threshold=at, variable_knn=vknn, drop_out=dropout, variable_dropout=vdropout)
                response = f'Your model has finished training with an accuracy of {DCA_accuracy * 100:.2f}% click save if you wish to keep it.'
                return Response(response,status=200, headers=HEADERS)
            else:
                model_name, dca_accuracy = main(loop, es, estats, knn, accuracy_threshold=at, variable_knn=vknn, drop_out=dropout, variable_dropout=vdropout)
                if dca_accuracy >= at:
                    response = f'Your model has finished training with an accuracy of {dca_accuracy * 100:.2f}% and was saved at {MODEL_SAVE_DIRECTORY}/{model_name}'
                else: 
                    response = f'Your model has finished training with an accuracy of {dca_accuracy * 100:.2f}% and was discarded'
                
                return Response(response,status=200, headers=HEADERS)
        else:
            model_name = save_model(model, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy,
                save_excel_stats=save_excel_stats, y_test_ages=y_test_ages, predicted=predicted, history=history, y_test=y_test, save_directory=save_directory)
            
            active_model.value = model_name
            

            model, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            response = f'Your model was saved at {MODEL_SAVE_DIRECTORY}/{model_name}'
            
            return Response(response,status=200, headers=HEADERS)


    except Exception as e:
        traceback.print_exc()
        return Response(status=500, headers=HEADERS)

    

if __name__=="__main__":

    #active_model = 'model8_accuracy_84.08'
    multiprocessing.set_start_method('spawn') 
    manager = Manager()
    active_model = manager.Value(c_char_p, "model9_accuracy_82.09")
    app.run(host='0.0.0.0',port=5000, threaded=True, use_reloader = False)