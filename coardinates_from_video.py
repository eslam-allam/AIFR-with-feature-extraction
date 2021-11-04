from typing import Text
import kivymd as k
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout

class Touch(Widget):

        def on_touch_down(self, touch):
            print(touch)
            #return super().on_touch_down(touch)
        
        def on_touch_move(self, touch):
            pass
            #return super().on_touch_move(touch)
        
        def on_touch_up(self, touch):
            pass
            #return super().on_touch_up(touch)

class mainAPP(MDApp):

    def build(self):
        layout = FloatLayout()
        self.image = Image()

        layout.add_widget(Touch())

        layout.add_widget(self.image)
        self.save_img_button = MDRaisedButton(
            text='CLICK HERE',
            pos_hint={'center_x': .5,'y': .01},
            size_hint=(0.02, 0.02)
        )
        
        self.save_img_button.bind(on_press=self.take_picture)
        
        layout.add_widget(self.save_img_button)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        
        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()

        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr', bufferfmt = 'ubyte')
        self.image.texture = texture
    
    def take_picture(self, *args):
        image_name = 'picture_at_thingy.png'
        cv2.imwrite(image_name, self.image_frame)

    

if __name__ == '__main__':
    mainAPP().run()