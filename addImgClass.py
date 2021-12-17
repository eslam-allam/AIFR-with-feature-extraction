from logging import exception
from math import e
import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import pandas as pd  # Import Pandas library
import sys  # Enables the passing of arguments
import os
from PIL import Image


cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

face_cascade = cv2.CascadeClassifier(cascPathface)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def swap_if_greater(x, y):
    if x > y:
        temp = x
        x = y
        y = temp
        return x, y
    return x, y


annotation_index = 0


class addImgClass:
    def __init__(self, imgPath):
        global annotation_index
        annotation_index = 0
        newPath = "./datasets/FGNET/newImages/"
        counter = 0
        print(imgPath)
        image = cv2.imread(imgPath[0])

        image = image_resize(image, height=600)
        image3 = image.copy()
        print(image.shape)

        # Create lists to store all x, y, and annotation values
        x_vals = []
        y_vals = []
        points = []
        annotation_vals = [
            "left_eye",
            "right_eye",
            "top_of_face",
            "bottom_of_face",
            "leftMost_of_face",
            "rightMost_0f_face",
        ]

        spacer = "-" * 20
        self.image2 = image.copy()

        def draw_circle(event, x, y, flags, param):
            global annotation_index

            if event == cv2.EVENT_LBUTTONDBLCLK:
                # Annotate the image
                txt = annotation_vals[annotation_index]

                # Append values to the list
                points.append((x, y))
                x_vals.append(x)
                y_vals.append(y)
                self.image2 = cv2.circle(self.image2, (x, y), 3, (0, 255, 0), -1)
                # Print the coordinates and the annotation to the console
                print(
                    "x = " + str(x) + "  y = " + str(y) + "  Annotation = " + txt + "\n"
                )

                # Prompt user for another annotation

                annotation_index += 1
                if annotation_index <= 5:
                    print(
                        "{}please double click on {}. Try to be as precise as possible!!{}".format(
                            spacer, annotation_vals[annotation_index], spacer
                        )
                    )

        print("Welcome to the Image Annotation Program!\n")
        print(
            "{}please double click on {}. Try to be as precise as possible!!{}".format(
                spacer, annotation_vals[0], spacer
            )
        )

        # We create a named window where the mouse callback will be established
        cv2.namedWindow("Image mouse")

        # We set the mouse callback function to 'draw_circle':
        cv2.setMouseCallback("Image mouse", draw_circle)
        try:

            count = 0
            while True:
                # Show image 'Image mouse':

                if count == 0:
                    cv2.imshow("Image mouse", image)
                    count = count + 1
                else:
                    cv2.imshow("Image mouse", self.image2)

                # Continue until 'q' is pressed:
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    print("Q KEY PRESSED!!!! QUITTING!!!!")
                    break
                if annotation_index > 5:
                    cv2.imshow("Image mouse", image)
                    print("ANNOTATION INDEX MORE THAN 5 !!!!! QUITTING!!")
                    break

            if not x_vals:
                raise Exception("NO POINTS ANNOTATED!!!!!EXITING")
            if len(x_vals) < 6:
                raise Exception("NUMBER OF ANNOTATED POINTS < 6!!!!EXITING")
            # Create a dictionary using lists
            data = {"X": x_vals, "Y": y_vals, "Annotation": annotation_vals}

            #%%
            left_eye = [x_vals[0], y_vals[0]]
            right_eye = [x_vals[1], y_vals[1]]

            up = y_vals[2]
            down = y_vals[3]
            left = x_vals[4]
            right = x_vals[5]

            # swap left and right eyes
            if right_eye[0] > left_eye[0]:
                temp = right_eye
                right_eye = left_eye
                left_eye = temp

            up, down = swap_if_greater(up, down)
            right, left = swap_if_greater(right, left)

            left_eye_x = left_eye[0]
            left_eye_y = left_eye[1]
            right_eye_x = right_eye[0]
            right_eye_y = right_eye[1]

            if left_eye_y > right_eye_y:
                A = (right_eye_x, left_eye_y)
                # Integer -1 indicates that the image will rotate in the clockwise direction
                direction = -1
                direction_text = "CLOCKWISE"
            else:
                A = (left_eye_x, right_eye_y)
                # Integer 1 indicates that image will rotate in the counter clockwise
                # direction
                direction = 1
                direction_text = "COUNTER CLOCKWISE"

            print(
                "left eye: {}\nright eye: {}\nROTATING {}".format(
                    left_eye, right_eye, direction_text
                )
            )

            delta_x = right_eye_x - left_eye_x
            delta_y = right_eye_y - left_eye_y
            angle = np.arctan(delta_y / delta_x)
            angle = (angle * 180) / np.pi

            # Width and height of the image
            h, w = image3.shape[:2]
            # Calculating a center point of the image
            # Integer division "//"" ensures that we receive whole numbers
            center = (w // 2, h // 2)
            # Defining a matrix M and calling
            # cv2.getRotationMatrix2D method
            M = cv2.getRotationMatrix2D(center, (angle), 1.0)
            # Applying the rotation to our image using the
            # cv2.warpAffine method
            rotated = cv2.warpAffine(image3, M, (w, h))

            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            print("shape of gray: ", gray.shape)
            print("{}:{},{}:{}".format(left, right, up, down))

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            gray = gray[up:down, right:left]

            print("shape of gray: ", gray.shape)
            # are we using clahe ??? doesn't have .apply
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ################

            equ = cv2.equalizeHist(gray)

            dim = (224, 224)
            resized = cv2.resize(equ, dim, interpolation=cv2.INTER_AREA)
            # print(resized.shape) should be deleted

            im = Image.fromarray(resized)
            annIM = Image.fromarray(self.image2)

            """cv2.imshow("img", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            # Create the Pandas DataFrame
            # df = pd.DataFrame(data)
            # print()
            # print(df)
            # print()

            # Export the dataframe to a csv file
            # df.to_csv(path_or_buf=csvPath, index=None, header=True)

            # Destroy all generated windows:
            cv2.destroyAllWindows()
            print("+++++++++++++++++++")
            print(annIM)
            print("+++++++++++++++++++")
            self.originalImagePath = imgPath
            self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
            self.annotImage = self.image2
            self.imgAfterPP = im
            self.csvFilePath = data
        except Exception as e:
            print(e)
            cv2.destroyAllWindows()
