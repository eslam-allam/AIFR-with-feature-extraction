# Necessary imports
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

FGnet_path = './datasets/FGNET/images/'
FGnet_points_path = './datasets/FGNET/points/'

images_path = os.listdir(FGnet_path)
imagePoints_path = os.listdir(FGnet_points_path)


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

cascPatheye = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye.xml"

face_cascade=cv2.CascadeClassifier(cascPathface)
eye_cascade=cv2.CascadeClassifier(cascPatheye)



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

    img = cv2.circle(img, left_eye_center, 2, (255, 0, 0), 2)
    img = cv2.circle(img, right_eye_center, 2, (255, 0, 0), 2)
    img = cv2.line(img,left_eye_center,right_eye_center,(0, 0, 255),1)

    #for x,y in points:
    #    img = cv2.circle(img, (x,y), 2, (255, 0, 0), 1)
    #chin ??
    chin = points[7]
    chinx = chin[0]
    chiny= chin[1]
    img = cv2.circle(img, (chinx,chiny), 2, (255, 0, 0), 1)

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

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(rotated,(x,y),(x+w,y+h),(255,0,0),2)
        temp = rotated[y:chiny, x:x + w]

        R, G, B = cv2.split(temp)

        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        equ = cv2.merge((output1_R, output1_G, output1_B))
        dim = (224, 224)
        resized = cv2.resize(equ, dim, interpolation = cv2.INTER_AREA)
        #print("hi")
        print(resized.shape)
        #tempAfterEq = cv2.equalizeHist(temp)

        #cv2.imshow("face",resized)
        #cv2.imwrite('face.jpg', resized)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    #numpy_horizontal = np.hstack((temp, resized))
    #print(resized.shape)
    #cv2.imshow("img",rotated)
    f, axarr = plt.subplots(2)
    
    axarr[0].imshow(rotated)
    axarr[1].imshow(resized)
    plt.show()
    #cv2.imshow("img",numpy_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()