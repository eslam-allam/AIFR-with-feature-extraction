import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout

from kivy.uix.button import Button

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.core.window import Window

from kivy.uix.scrollview import ScrollView


import os
import math

from addImgClass import addImgClass

"""class BoxLayoutExample(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        b1 = Button(text="A")
        b2 = Button(text="B")
        b3 = Button(text="C")
        self.add_widget(b1)
        self.add_widget(b2)
        self.add_widget(b3)"""

from plyer import filechooser


class MainWindow(Screen):
    pass


class SecondWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        firstArea = BoxLayout(orientation="vertical")

        toolBar = BoxLayout(orientation="horizontal")
        toolBar.size_hint = (1, None)
        toolBar.height = "60dp"

        backButton = Button(
            size_hint=(None, None),
            pos_hint={"x": 0, "center_y": 0.5},
            width="100dp",
            height="40dp",
            text="go back",
        )

        backButton.bind(on_press=self.callback)

        scrollTry = ScrollView(pos_hint={"center_x": 0.5, "center_y": 0.5}, height=5000)
        print(self.size)
        print("self.size")
        theImagesGrid = StackLayout()

        FGnet_path = "./datasets/FGNET/newImages/"
        images_path = os.listdir(FGnet_path)
        sizz = 0
        for i in enumerate(images_path):
            sizz = sizz + 1

        hhh = math.ceil(sizz / 4) * 200
        grid = GridLayout(
            size_hint_y=None,
            cols=4,
            row_default_height="200dp",
            row_force_default=True,
            spacing=(0, 0),
            padding=(0, 0),
            height=str(hhh) + "dp",
        )

        for i, path in enumerate(images_path):
            img = Image(source=(FGnet_path + path), size_hint=(0.2, 0.2))
            grid.add_widget(img)
            # if i == 100:
            #   break

        """for i in range(5):
            x = Label(text=str(i))
            theImagesGrid.add_widget(x)"""

        """b1 = Button(text="b1")
        b2 = Button(text="b1")
        b3 = Button(text="b1")
        theImagesGrid.add_widget(b1)
        theImagesGrid.add_widget(b2)
        theImagesGrid.add_widget(b3)"""

        toolBar.add_widget(backButton)
        scrollTry.add_widget(grid)

        firstArea.add_widget(toolBar)
        firstArea.add_widget(scrollTry)

        self.add_widget(firstArea)

    def callback(self, instance):
        print("Button is pressed")
        print("The button % s state is <%s>" % (instance, instance.state))
        self.manager.current = "main"
        self.manager.transition.direction = "right"


class thirdWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        box = BoxLayout(
            orientation="vertical",
            padding=(10, 10),
            # row_default_height="48dp",
            # row_force_default=True,
            spacing=(10, 10),
        )

        uploadB = Button(text="upload")
        uploadB.bind(on_release=self.upload)
        box.add_widget(uploadB)
        self.add_widget(box)

    def upload(self, instance):
        path = filechooser.open_file(title="Pick a img file..")
        csvPath = filechooser.open_file(title="Pick a csv file..")
        addImgClass(path, csvPath)


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("guiCode.kv")


class myTryApp(App):
    def build(self):

        # Window.clearcolor = (1, 1, 1, 1)
        return kv


if __name__ == "__main__":
    myTryApp().run()
