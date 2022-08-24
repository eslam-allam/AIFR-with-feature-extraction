import base64
from flask import Flask, Response, render_template, request
import cv2
import sys
from collections import deque
import numpy as np

sys.path.append('../..')
from mediapipecam import face_mesh
from allign import proccess_image




app = Flask(__name__)

#camera=cv2.VideoCapture(2)

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
    
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame', status=200, headers={'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'})

@app.route('/capture', methods=['GET', 'POST', 'OPTIONS'])
def capture():

    global current_pic
    global name

    args = request.args
    save = args.get('save')
    get_result = args.get('getresult') 
   
    if get_result == 'True':
        return Response(cv2.imencode('.jpg', current_pic)[1].tobytes(),status=200, headers={'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'})


    
    if save == 'False':
        if request.method == 'GET':
            image=frames.pop()
            name = False
        else:
            
            image = request.files.get('files.myImage')
            name = image.filename
            
            
            image = np.fromfile(image)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            

        image = proccess_image(image)
        current_pic = image

        if image is None: 
            print('face not found')
            return Response(status=403, headers={'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade"})
        image = cv2.imencode('.jpg', image)[1].tobytes()

        return Response(image,status=200, headers={'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'})
    else:
        if name: cv2.imwrite(f'../../test_single_image/{name}', current_pic)
        else:
            cv2.imwrite('../../test_single_image/test.jpg', current_pic)
        return Response(status=200, headers={'Access-Control-Allow-Origin': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade","content-type":'image/jpeg'})
    
@app.route('/processdirectory', methods=['GET'])
def processdirectory():
    args = request.args
    print(args)

    
    
    return Response(status=200, headers={'Access-Control-Allow-Origin ': "*", "Access-Control-Allow-Headers":'Content-Type', 'Referrer-Policy':"no-referrer-when-downgrade"})

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000, threaded=True, use_reloader = False)