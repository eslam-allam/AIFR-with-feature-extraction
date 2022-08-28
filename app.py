
import os
from flask import Flask, Response, render_template, request
import cv2
from collections import deque
import numpy as np
from mediapipecam import face_mesh
from allign import proccess_image
from AIFR_VGG import MODEL_SAVE_DIRECTORY, main, save_model
import logging
import traceback


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



model, m1, m2, flatten, Ax1, Ax2, Ax3, Ay1, Ay2, Ay3, classifier, DCA_accuracy, save_excel_stats, y_test_ages, predicted, history, y_test, save_directory = None, None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None

NEW_IMAGE_DIRECTORY = '/home/eslamallam/Python/AIFR-with-feature-extraction/datasets/FGNET/newImages'
HEADERS = {'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'}



app = Flask(__name__)

#camera=cv2.VideoCapture(2)

def string_to_bool(expression : str):
    if expression.casefold() == 'true':
        return True
    else: return False

frames = deque(maxlen=20)
current_pic = None
name = False

def generate_frames():
    camera = cv2.VideoCapture(0)  
    while True:
         
        ## read the camera frame
        success,frame=camera.read()
        frames.append(frame)
        frame = face_mesh(frame)
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()


        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        #time.sleep(0.1)
    camera.release()
    
    



@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/video')
def video():
    
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame', status=200, headers=HEADERS)

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
    knn = int(args['knn'])
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