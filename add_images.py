#%%
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import pandas as pd # Import Pandas library
import sys # Enables the passing of arguments
import os
from PIL import Image


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

face_cascade=cv2.CascadeClassifier(cascPathface)


# Project: Annotate Images Using OpenCV
# Author: Addison Sears-Collins
# Date created: 9/11/2019
# Python version: 3.7
# Description: This program allows users to click in an image, annotate a 
#   number of points within an image, and export the annotated points into
#   a CSV file.
newPath ='./datasets/FGNET/newImages/'
counter = 0

# Define the file name of the image
INPUT_IMAGE = 'new_images/'+sys.argv[1] # "cat_dog.jpg"
IMAGE_NAME = INPUT_IMAGE[:INPUT_IMAGE.index(".")]
OUTPUT_IMAGE = IMAGE_NAME + "_annotated.jpg"
output_csv_file = sys.argv[2]

# Load the image and store into a variable
# -1 means load unchanged
image = cv2.imread(INPUT_IMAGE)
print(image.shape)


# Create lists to store all x, y, and annotation values
x_vals = []
y_vals = []
points = []
annotation_vals = ['left_eye','right_eye','top_of_face','bottom_of_face','leftMost_of_face','rightMost_0f_face']

annotation_index = 0
spacer = '-'*20

def draw_circle(event, x, y, flags, param):
    global annotation_index

    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # Annotate the image
        txt = annotation_vals[annotation_index]
        

        # Append values to the list
        points.append((x,y))
        x_vals.append(x)
        y_vals.append(y)

        # Print the coordinates and the annotation to the console
        print("x = " + str(x) + "  y = " + str(y) + "  Annotation = " + txt + "\n")

        # Prompt user for another annotation
        
        annotation_index += 1
        if annotation_index <= 5:
            print("{}please double click on {}. Try to be as precise as possible!!{}".format(spacer,annotation_vals[annotation_index],spacer))

        

print("Welcome to the Image Annotation Program!\n")
print("{}please double click on {}. Try to be as precise as possible!!{}".format(spacer,annotation_vals[0],spacer))

# We create a named window where the mouse callback will be established
cv2.namedWindow('Image mouse')

# We set the mouse callback function to 'draw_circle':
cv2.setMouseCallback('Image mouse', draw_circle)

while True:
    # Show image 'Image mouse':
    print('annotation index: {}'.format(annotation_index))
    cv2.imshow('Image mouse', image)

    # Continue until 'q' is pressed:
    if cv2.waitKey(20) & 0xFF == ord('q'):
        
        break
    if annotation_index > 5:
        break
cv2.destroyAllWindows()
# Create a dictionary using lists
data = {'X':x_vals,'Y':y_vals,'Annotation':annotation_vals}

#%%

left_eye_x = x_vals[0] 
left_eye_y = y_vals[0]


right_eye_x = x_vals[1]
right_eye_y = y_vals[1]


up = y_vals[2]
down = y_vals[3]
left = x_vals[4]
right = x_vals[5]

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
h, w = image.shape[:2]
# Calculating a center point of the image
# Integer division "//"" ensures that we receive whole numbers
center = (w // 2, h // 2)
# Defining a matrix M and calling
# cv2.getRotationMatrix2D method
M = cv2.getRotationMatrix2D(center, direction*(angle), 1.0)
# Applying the rotation to our image using the
# cv2.warpAffine method
rotated = cv2.warpAffine(image, M, (w, h))

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
print("shape of gray: ",gray.shape)
print("{}:{},{}:{}".format(left,right,up,down))

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

gray = gray[up:down,left:right]

print("shape of gray: ",gray.shape)
# are we using clahe ??? doesn't have .apply 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
################ 

equ = cv2.equalizeHist(gray)

dim = (224, 224)
resized = cv2.resize(equ, dim, interpolation = cv2.INTER_AREA)
#print(resized.shape) should be deleted



im = Image.fromarray(resized)

cv2.imshow("img",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Create the Pandas DataFrame
df = pd.DataFrame(data)
print()
print(df)
print()

# Export the dataframe to a csv file
df.to_csv(path_or_buf = output_csv_file, index = None, header=True) 

# Destroy all generated windows:
cv2.destroyAllWindows()