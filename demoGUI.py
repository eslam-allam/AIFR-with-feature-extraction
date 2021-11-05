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
import re

import os
import math

from numpy.lib.utils import source

from addImgClass import addImgClass


import pyautogui

Window.size = (2000, 1000)
Window.top = int((pyautogui.size().height - Window.height)) / 2
Window.left = int((pyautogui.size().width - Window.width)) / 2

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

from kivy.config import Config


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
        super(thirdWindow, self).__init__(**kwargs)

        self.imgsToSave = []
        self.selectedImg = None

        box = BoxLayout(
            orientation="vertical",
            padding=(10, 10),
            # row_default_height="48dp",
            # row_force_default=True,
            spacing=(10, 10),
        )

        toolBar = BoxLayout(orientation="horizontal")
        toolBar.size_hint = (1, None)
        toolBar.height = "60dp"

        ######################## display area ############################################################################## might need to add a remove to avoid any bugs
        self.midBox = BoxLayout(orientation="horizontal")
        if self.selectedImg is None:
            placeHolderForimg = Label(text="placeHolderForimg")
            self.midBox.add_widget(placeHolderForimg)
        else:
            pass
            # placeHolderForimg = Image(source=self.imgsToSave[self.selectedImg][0])
            # self.midBox.add_widget(placeHolderForimg)

        #####################################################################################################################
        uploadBox = BoxLayout(orientation="vertical", size_hint=(0.2, 1))

        uploadB = Button(
            text="upload",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint=(0.9, 1),
        )
        uploadB.bind(on_release=self.upload)

        uploadBox.add_widget(uploadB)

        self.bottomBar = BoxLayout(orientation="horizontal", size_hint=(1, 0.15))

        scrollTry = ScrollView(pos_hint={"center_x": 0.5, "center_y": 0.5})

        FGnet_path = "./forDisplayTry/"
        images_path = os.listdir(FGnet_path)

        # the temp display
        """ counter69 = 0  # must be a method for this
        for i in enumerate(images_path):
            counter69 = counter69 + 1

        scrollLength = (counter69 * 100) + ((counter69 - 1) * 20)"""

        scrollCounter = 0
        for i in self.imgsToSave:
            scrollCounter = scrollCounter + 1
        scrollLength = (scrollCounter * 100) + ((scrollCounter - 1) * 20)

        self.leftSide = GridLayout(
            size_hint_x=None,
            rows=1,
            col_default_width="100dp",
            col_force_default=True,
            spacing=("20dp", 0),
            padding=(0, 0),
            width=str(scrollLength) + "dp",
        )

        # the temp display
        """for i, path in enumerate(images_path):
            tempButton = Button(background_normal=FGnet_path + path)
            self.leftSide.add_widget(tempButton)"""

        for idx, val in enumerate(
            self.imgsToSave
        ):  # useless kinda might remove it later since it always starts empty
            tempButton = Button(background_normal=val[0][0])
            tempButton.bind(
                on_press=lambda x: self.disp(x=idx)
            )  # on_press=lambda x: self.on_press(x=sum)
            self.leftSide.add_widget(tempButton)

        scrollTry.add_widget(self.leftSide)

        self.rightSide = BoxLayout(orientation="vertical", size_hint=(0.2, 1))

        confirm = Button(
            text="confirm",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint=(0.9, 1),
        )
        addPoints = Button(
            text="Update points",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint=(0.9, 1),
        )

        self.rightSide.add_widget(confirm)
        self.rightSide.add_widget(addPoints)

        self.bottomBar.add_widget(uploadBox)
        self.bottomBar.add_widget(scrollTry)
        self.bottomBar.add_widget(self.rightSide)

        backButton = Button(
            size_hint=(None, None),
            pos_hint={"x": 0, "center_y": 0.5},
            width="100dp",
            height="40dp",
            text="go back",
        )

        backButton.bind(on_press=self.callback)
        toolBar.add_widget(backButton)
        box.add_widget(toolBar)
        box.add_widget(self.midBox)
        box.add_widget(self.bottomBar)
        # box.add_widget(uploadB)  #####################
        self.add_widget(box)

    def upload(self, instance):
        path = filechooser.open_file(title="Pick a img file..")
        if not path:
            print("{}\nNO IMAGE CHOSEN!!!!\n{}".format("-" * 10, "-" * 10))
            return False

        image_name = re.findall("\w+\.\w+", path[0])
        image_name = image_name[0]
        csvPath = re.sub(image_name[-4:], ".csv", path[0])
        print(path, "\n", csvPath)

        values = addImgClass(path, csvPath)

        listOfValues = [values.originalImagePath, values.imgAfterPP, values.csvFilePath]
        print(listOfValues)
        self.imgsToSave.append(listOfValues)
        self.imgUpdate()

    def callback(self, instance):
        print("Button is pressed")
        print("The button % s state is <%s>" % (instance, instance.state))
        self.manager.current = "main"
        self.manager.transition.direction = "right"

    def imgUpdate(self):
        print("i am at update")
        self.removeBeforeUpdate()
        for idx, val in enumerate(self.imgsToSave):
            tempButton = Button(background_normal=val[0][0])
            tempButton.bind(on_press=lambda x: self.disp(x=idx))
            self.leftSide.add_widget(tempButton)

    def removeBeforeUpdate(self):
        rows = [i for i in self.leftSide.children]
        for row1 in rows:
            self.leftSide.remove_widget(row1)

    def removeMidBox(self):
        rows = [i for i in self.midBox.children]
        for row1 in rows:
            self.midBox.remove_widget(row1)

    def disp(self, x):
        self.removeMidBox()
        print(x)
        self.selectedImg = x
        for idx, val in enumerate(self.imgsToSave):
            if idx == self.selectedImg:
                placeHolderForimg = Image(source=val[0][0])
                self.midBox.add_widget(placeHolderForimg)


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("guiCode.kv")


class myTryApp(App):
    def build(self):

        # Window.clearcolor = (1, 1, 1, 1)
        return kv


if __name__ == "__main__":

    myTryApp().run()
