# install and import above modules first
import os
import cv2
import math
import matplotlib.pyplot as pl
import pandas as pd
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as normalizeCoardinates


# Detect face
def face_detection(img, thickness=2):
	img = cv2.resize(img, (224, 224))
	faces = detector.detect(img)
	assert faces[1] is not None, 'Cannot find face in image'
	if faces[1] is None:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
	else:
		coords = faces[1][0][:-1].astype(np.int32)
		top_left, width, hight, score = coords[0:2], coords[2], coords[3], coords[-1]
		'''cv2.circle(img, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
		cv2.circle(img, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
		cv2.circle(img, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
		cv2.circle(img, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
		cv2.circle(img, (coords[12], coords[13]), 2, (0, 255, 255), thickness)'''
		img = img[top_left[1]:top_left[1]+hight, top_left[0]:top_left[0]+width]
		return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def trignometry_for_distance(a, b):
	return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
					((b[1] - a[1]) * (b[1] - a[1])))

# Find eyes
def Face_Alignment(img_path):
	img_raw = cv2.imread(img_path).copy()
	img, gray_img = face_detection(cv2.imread(img_path))
	gray_img = process_image(gray_img)
	eyes = eye_detector.detectMultiScale(gray_img)
	

	# for multiple people in an image find the largest
	# pair of eyes
	if len(eyes) >= 2:
		eye = eyes[:, 2]
		container1 = []
		for i in range(0, len(eye)):
			container = (eye[i], i)
			container1.append(container)
		df = pd.DataFrame(container1, columns=[
						"length", "idx"]).sort_values(by=['length'])
		eyes = eyes[df.idx.values[0:2]]

		# deciding to choose left and right eye
		eye_1 = eyes[0]
		eye_2 = eyes[1]
		if eye_1[0] > eye_2[0]:
			left_eye = eye_2
			right_eye = eye_1
		else:
			left_eye = eye_1
			right_eye = eye_2

		# center of eyes
		# center of right eye
		right_eye_center = (
			int(right_eye[0] + (right_eye[2]/2)),
		int(right_eye[1] + (right_eye[3]/2)))
		right_eye_x = right_eye_center[0]
		right_eye_y = right_eye_center[1]
		cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

		# center of left eye
		left_eye_center = (
			int(left_eye[0] + (left_eye[2] / 2)),
		int(left_eye[1] + (left_eye[3] / 2)))
		left_eye_x = left_eye_center[0]
		left_eye_y = left_eye_center[1]
		cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

		# finding rotation direction
		if left_eye_y > right_eye_y:
			print("Rotate image to clock direction")
			point_3rd = (right_eye_x, left_eye_y)
			direction = -1 # rotate image direction to clock
		else:
			print("Rotate to inverse clock direction")
			point_3rd = (left_eye_x, right_eye_y)
			direction = 1 # rotate inverse direction of clock

		cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
		a = trignometry_for_distance(left_eye_center,
									point_3rd)
		b = trignometry_for_distance(right_eye_center,
									point_3rd)
		c = trignometry_for_distance(right_eye_center,
									left_eye_center)
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = (np.arccos(cos_a) * 180) / math.pi

		if direction == -1:
			angle = 90 - angle
		else:
			angle = -(90-angle)

		# rotate image
		new_img = Image.fromarray(img_raw)
		new_img = np.array(new_img.rotate(direction * angle))

	return new_img

def process_image(img):
    #dim = (224, 224)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return gray
    


opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]
path = folders[0]
for folder in folders[1:]:
	path = path + "/" + folder
path_for_face = path+"/data/haarcascade_frontalface_alt2.xml"
path_for_eyes = path+"/data/haarcascade_eye.xml"
path_for_nose = path+"/data/haarcascade_mcs_nose.xml"

modelFile = "models/face_detection_yunet_2022mar-act_int8-wt_int8-quantized.onnx"
detector = cv2.FaceDetectorYN.create(
        modelFile,
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
detector.setInputSize((224, 224))

face_detector = cv2.CascadeClassifier(path_for_face)
eye_detector = cv2.CascadeClassifier(path_for_eyes)

mp_face_detection = mp.solutions.face_detection
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # model selection
    min_detection_confidence=0.5  # confidence threshold
)



# Name of the image for face alignment if on
# the other folder kindly paste the name of
# the image with path included
test_set = ["./datasets/FGNET/images/001A16.JPG"]
for i in test_set:
    alignedFace = Face_Alignment(i)
    pl.imshow(alignedFace[:, :, ::-1])
    pl.show()
    img, gray_img = face_detection(alignedFace)
    pl.imshow(img[:, :, ::-1])
    pl.show()
    img = process_image(img)
    pl.imshow(img,cmap='gray')
    pl.show()
