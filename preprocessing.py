import os
import cv2 as cv
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

FGnet_path = './datasets/FGNET/images/'
FGnet_points_path = './datasets/FGNET/points/'
cascPathface = os.path.dirname(
    cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv.CascadeClassifier(cascPathface)


images_path = os.listdir(FGnet_path)
imagePoints_path = os.listdir(FGnet_points_path)

for i,path in enumerate(images_path):
    img = cv.imread(FGnet_path+path,0)
    cl1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cl1.apply(img)
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    points = np.loadtxt(FGnet_points_path+imagePoints_path[i],comments=("version:", "n_points:", "{", "}"))
    points = points.astype('int32')
    

    faces = faceCascade.detectMultiScale(img,
                                        scaleFactor=1.1,
                                        minNeighbors=4,
                                        minSize=(60, 60),
                                        flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h),(0,255,0), 2)


    #img = cv.circle(img, (points[31]), 2, (255, 0, 0), 1)
    #img = cv.circle(img, (points[36]), 2, (255, 0, 0), 1)
    for x,y in points:
        img = cv.circle(img, (x,y), 2, (255, 0, 0), 1)
    
    
    print('face coardinates: {},{}:{},{}'.format(x,y,x+w,y+y))

    

    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()





