__author__ = 'bunkus'
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp
from allign import proccess_image
import os
import sys
from kivy.uix.button import Button


PROCESSED_IMAGE_DIRECTORY = './test_single_image/output'



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


class CamApp(App):
    
    def process_and_save_image(self, instance):
        processed_image = proccess_image(self.image_cap)

        if not os.path.exists(PROCESSED_IMAGE_DIRECTORY): os.mkdir(PROCESSED_IMAGE_DIRECTORY)
        image_names = os.listdir(PROCESSED_IMAGE_DIRECTORY)

        if not image_names: name = 'processed_capture_0.jpg'
        else:
            image_names = [name for name in image_names if 'processed_capture_' in name] 
            name = f'processed_capture_{len(image_names)}.jpg'
        
        path = PROCESSED_IMAGE_DIRECTORY+'/'+name
        cv2.imwrite(path, processed_image)
        print(path)

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        
        button = Button(text="Capture", size=(200,100), size_hint= (None, None))
        button.bind(on_press=self.process_and_save_image)

        layout.add_widget(self.img1)
        layout.add_widget(button)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        success, image = self.capture.read()
        
        
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
           
            self.image_cap = image.copy()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            
            
        # convert it to texture
        buf1 = cv2.flip(image, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

        return image

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()