# IMPORTING LIBRARIES
from asyncore import read
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm as meter
import sys




# INITIALIZING OBJECTS
mp_face_mesh = mp.solutions.face_mesh

FINAL_DIMENSIONS = (224, 224)
DATASET_DIRECTORY = './datasets/FGNET/images'
PROCESSED_IMAGE_DIRECTORY = './datasets/FGNET/newImages'




def proccess_image(image):
    # DETECT THE FACE LANDMARKS
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    
        
        shape = image.shape

        # convert the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results = face_mesh.process(image) 


        # To improve performance
        image.flags.writeable = True

        # Convert back to the BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        face_landmarks = results.multi_face_landmarks[0]

        left_eye_landmarks = [
        face_landmarks.landmark[35] , face_landmarks.landmark[246], face_landmarks.landmark[161], face_landmarks.landmark[160],
        face_landmarks.landmark[159], face_landmarks.landmark[158], face_landmarks.landmark[157], face_landmarks.landmark[173], 
        face_landmarks.landmark[133], face_landmarks.landmark[155], face_landmarks.landmark[154], face_landmarks.landmark[153],
        face_landmarks.landmark[145], face_landmarks.landmark[144], face_landmarks.landmark[163], face_landmarks.landmark[7] ]

        x = [(p.x)*shape[1] for p in left_eye_landmarks]
        y = [(p.y)*shape[0] for p in left_eye_landmarks]

        left_eye = (sum(x) / len(left_eye_landmarks), sum(y) / len(left_eye_landmarks))
        left_eye = [int(p) for p in left_eye]
        left_eye_x = left_eye[0]
        left_eye_y = left_eye[1]

        right_eye_landmarks = [
        face_landmarks.landmark[362], face_landmarks.landmark[398], face_landmarks.landmark[384], face_landmarks.landmark[385],
        face_landmarks.landmark[386], face_landmarks.landmark[387], face_landmarks.landmark[388], face_landmarks.landmark[466], 
        face_landmarks.landmark[263], face_landmarks.landmark[249], face_landmarks.landmark[390], face_landmarks.landmark[373],
        face_landmarks.landmark[374], face_landmarks.landmark[380], face_landmarks.landmark[381], face_landmarks.landmark[382]]

        x = [(p.x)*shape[1] for p in right_eye_landmarks]
        y = [(p.y)*shape[0] for p in right_eye_landmarks]

        right_eye = (sum(x) / len(right_eye_landmarks), sum(y) / len(right_eye_landmarks))
        right_eye = [int(p) for p in right_eye]
        right_eye_x = right_eye[0]
        right_eye_y = right_eye[1]

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
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the
        # cv2.warpAffine method
        image = cv2.warpAffine(image, M, (w, h))

        
        # To improve performance
        image.flags.writeable = False

        results = face_mesh.process(image) 

        # To improve performance
        image.flags.writeable = True

        face_landmarks = results.multi_face_landmarks[0] 

        face_landmarks = [p for p in face_landmarks.landmark] 

        x = [int(p.x * shape[1]) for p in face_landmarks]
        y = [int(p.y * shape[0]) for p in face_landmarks]

        mid_left = np.min(x)
        mid_right = np.max(x)
        mid_top = np.min(y)
        mid_bot = np.max(y)

        image = image[mid_top:mid_bot, mid_left:mid_right]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = cv2.resize(image, FINAL_DIMENSIONS, interpolation = cv2.INTER_AREA)

        return image


def main(single_image=False, dataset_directory=DATASET_DIRECTORY, processed_image_directory=PROCESSED_IMAGE_DIRECTORY, image_name=None):

    if not single_image:
        images_list = os.listdir(dataset_directory)

        for name in meter(images_list):
            path = dataset_directory+'/'+name
            new_path = processed_image_directory+'/'+name
            image = cv2.imread(path).copy()
            image = proccess_image(image)

            if not os.path.exists(processed_image_directory): os.mkdir(processed_image_directory)
                
            cv2.imwrite(new_path, image)
    else:
            assert image_name, 'Single image mode requires passing "image_name" argument.'
            path = dataset_directory+'/'+image_name
            new_path = processed_image_directory+'/'+image_name
            image = cv2.imread(path).copy()
            image = proccess_image(image)

            if not os.path.exists(processed_image_directory): os.mkdir(processed_image_directory)
                
            cv2.imwrite(new_path, image)

args = sys.argv

assert not('--one-image' in args and '--from-directory' in args), 'Cannot use one-image and from-directory modes at the same time'

if '--cli' in args:
    assert '--one-image' in args or '--from-directory' in args, 'CLI mode must be used with either --one-image or --from-directory.'
    
    src_directory = input(f'Enter source image directory excluding filename(If empty {DATASET_DIRECTORY} will be used): ')
    if not src_directory: src_directory = DATASET_DIRECTORY

    out_directory = input(f'Enter output directory excluding file name (if empty {PROCESSED_IMAGE_DIRECTORY} will be used): ')
    if not out_directory: out_directory = PROCESSED_IMAGE_DIRECTORY

    if '--one-image' in args:
        name = input('Enter image name (This field is mandatory): ')
        assert name, 'You must provide a name for the image'
        main(single_image=True, dataset_directory=src_directory, processed_image_directory=out_directory, image_name=name)

    elif '--from-directory' in args:
        main(dataset_directory=src_directory, processed_image_directory=out_directory)

        

        

