from itertools import count
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

FGnet_path = './datasets/FGNET/images/'
newPath ='./datasets/FGNET/newImages/'
FGnet_points_path = './datasets/FGNET/points/'

images_path = os.listdir(FGnet_path)
imagePoints_path = os.listdir(FGnet_points_path)


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

cascPatheye = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye.xml"

face_cascade=cv2.CascadeClassifier(cascPathface)
eye_cascade=cv2.CascadeClassifier(cascPatheye)

counter = 0
ess = []
allFacesRotated = []
for i,path in enumerate(images_path):
    
    img = cv2.imread(FGnet_path+path)
    
    points = np.loadtxt(FGnet_points_path+imagePoints_path[i],comments=("version:", "n_points:", "{", "}"))
    points = points.astype('int32')

    # Calculating coordinates of a central points of the rectangles
    left_eye_center = points[31]
    left_eye_x = left_eye_center[0] 
    left_eye_y = left_eye_center[1]

    right_eye_center = points[36]
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

   

    
    chin = points[7]
    chinx = chin[0]
    chiny= chin[1]
   
    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1 
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise  
        # direction
        direction = 1 


    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated = cv2.warpAffine(img, M, (w, h))


    
    allFacesRotated.append(rotated)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    right = points[0][0]
    left = points[0][0]
    up = points[0][1]
    down = points[0][1]
    for x in range(64):
        if(right<points[x][0]):
            right = points[x][0]
        if(left>points[x][0]):
            left = points[x][0]
        if(up>points[x][1]):
            up = points[x][1]
        if(down<points[x][1]):
            down = points[x][1]
    #print(faces)
    if(faces == ()):
        
            
        gray = gray[up-50:down, left:right]
        counter = counter + 1
        
    else:
        for (x,y,w,h) in faces:
            counter = counter + 1
        
            cv2.rectangle(rotated,(x,y),(x+w,y+h),(255,0,0),2)
        
            gray = gray[y:down, left:right]
     
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equ = clahe.apply(gray)
    dim = (224, 224)
    resized = cv2.resize(equ, dim, interpolation = cv2.INTER_AREA)
    print(resized.shape)


    print(type(resized))
    im = Image.fromarray(resized)
    newName = path.split("A")
    
    im.save(newPath+path)
    cv2.imshow('img',resized) #remove if u want to save all imgs
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   
