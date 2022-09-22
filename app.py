from multiprocessing import Process , Manager, shared_memory
from multiprocessing.sharedctypes import Value



import os
from flask import Flask, Response, render_template, request
import cv2
import numpy as np
from mediapipecam import face_mesh
from allign import proccess_image
from AIFR_VGG import  load_model, main, save_model
import AIFR_VGG
import logging
import traceback
from keras_vggface import utils



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



#active_model, active_Ax1, active_Ax2, active_Ax3, active_Ay1, active_Ay2, active_Ay3, active_classifier= load_model(model_name='model7_accuracy_89.55_enhanced')

NEW_IMAGE_DIRECTORY = '/home/eslamallam/Python/AIFR-with-feature-extraction/datasets/FGNET/newImages'
HEADERS = {'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'}

video_error = cv2.imread('guiImages/video_error.png')
ret, video_error = cv2.imencode('.jpg',video_error)
video_error = video_error.tobytes()

app = Flask(__name__)

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

manager = Manager()


current_pic = None
name = False


def generate_frames():
    
    shm = shared_memory.SharedMemory(create=True, size=1150528, name='processed_frame_memory')
    shared_processed_frame = np.ndarray((224, 224, 3), dtype='int32', buffer=shm.buf)
    prediction = Value('i', 0)
    parallel = Process(target= AIFR_VGG.predict, args=(prediction,))
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
                processed_frame = np.repeat(processed_frame[..., np.newaxis], 3, -1)
                if processed_frame is None: continue
                
                
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
   
    

    
            
      
    
    
    
@app.route("/predict")
def predict():
    frame = frames[-1]
    frame = proccess_image(frame)
    frame =  utils.preprocess_input(frame,version=2)
    fc1 = m1.predict(frame)
    fc2 = m2.predict(frame)
    pooling = pooling.predict(frame)

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

    print(f'prediction: {predicted}')


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/video')
def video():
   
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame', headers=HEADERS)
    
    

@app.route('/capture', methods=['GET', 'POST', 'OPTIONS'])
def capture():

    global current_pic
    global name
    
    new_image_directory = ''
    args = request.args
    save = args.get('save')
    get_result = args.get('getresult') 
    name = args.get('name').lower()
    age = args.get('age')
    
   
    if get_result == 'True':
        return Response(cv2.imencode('.jpg', current_pic)[1].tobytes(),status=200, headers=HEADERS)


    try:
        if save == 'False':
            if request.method == 'GET':
                image=frames.pop()
                name = False
            else:
                
                image = request.files.get('files.myImage')
                

                
                
                
                image = np.fromfile(image)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                

            image = proccess_image(image)
            current_pic = image

            if image is None: 
                print('face not found')
                return Response(status=403, headers=HEADERS)
            image = cv2.imencode('.jpg', image)[1].tobytes()
            

            return Response(image,status=200, headers=HEADERS)
        else:
            
            last_image_name = sorted(os.listdir(NEW_IMAGE_DIRECTORY))[-1]
            last_image_name = last_image_name.split('.')[0].split('A')[0].split('-')
            last_id, last_name = int(last_image_name[0]), last_image_name[1].lower()
            
            if last_name != name: last_id = last_id + 1
            
            name = f'{str(last_id).zfill(3)}-{name}A{age.zfill(2)}.jpg'
            
            if new_image_directory: cv2.imwrite(f'{new_image_directory}/{name}', current_pic)
            else:
                cv2.imwrite(f'{NEW_IMAGE_DIRECTORY}/{name}', current_pic)
            
            return Response(status=200, headers=HEADERS)
    except:
        return Response(status=500, headers=HEADERS)
    
@app.route('/trainmodel', methods=['GET'])
def trainmodel():
    global model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory

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
                model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory = main(loop, es, estats, knn, accuracy_threshold=at, variable_knn=vknn, drop_out=dropout, variable_dropout=vdropout)
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
            model_name = save_model(model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy,
                save_excel_stats=save_excel_stats, y_test_ages=y_test_ages, predicted=predicted, history=history, y_test=y_test, save_directory=save_directory)
            response = f'Your model has finished training with an accuracy of {DCA_accuracy * 100:.2f}% and was saved at {MODEL_SAVE_DIRECTORY}/{model_name}'
            return Response(response,status=200, headers=HEADERS)


    except Exception as e:
        traceback.print_exc()
        return Response(status=500, headers=HEADERS)

    

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000, threaded=True, use_reloader = False)