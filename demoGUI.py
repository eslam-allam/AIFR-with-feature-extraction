import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout

from kivy.uix.button import Button
from kivy.uix.popup import Popup

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

from io import BytesIO

from PIL import Image as PILImage
from kivy.core.image import Image as Core_Image
from kivy.uix.image import Image as kiImage


import pyautogui
import cv2
from matplotlib import pyplot as plt
import pandas as pd  # Import Pandas library

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

currDirectory = os.getcwd()

# os.chdir(currDirectory)  # reset the directory (add it it when do u want to reset a directory)


class MainWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        backgroundGrid = BoxLayout(orientation="horizontal")

        # left side grid

        leftSideGrid = BoxLayout(
            orientation="vertical",
            size_hint=(1, 0.9),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        mainPageImg = Image(
            source=("./guiImages/facial-recognition-connected-real-estate.png"),
            size_hint=(0.9, 1),
            pos_hint={"center_x": 0.5},
        )

        leftTopic = Label(
            size_hint=(1, 0.1),
            font_size="30sp",
            text="Age inavraint face recognition demo",
        )

        leftNames = Label(
            size_hint=(1, 0.1),
            text="Project created by Moataz Shaker and Eslam Allam",
        )

        leftSideGrid.add_widget(mainPageImg)
        leftSideGrid.add_widget(leftTopic)
        leftSideGrid.add_widget(leftNames)

        # right side grid

        rightSideGrid = BoxLayout(
            orientation="vertical",
            size_hint=(1, 0.8),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            spacing="10dp",
        )

        but1 = Button(
            text="Display dataset images",
            font_size="30sp",
            size_hint=(0.8, 0.9),
            pos_hint={"center_x": 0.5},
        )
        but1.bind(on_press=self.but1Call)

        but2 = Button(
            text="Add new set of images",
            font_size="30sp",
            size_hint=(0.8, 0.9),
            pos_hint={"center_x": 0.5},
        )
        but2.bind(on_press=self.but2Call)

        but3 = Button(
            text="Re train the dnn",
            font_size="30sp",
            size_hint=(0.8, 0.9),
            pos_hint={"center_x": 0.5},
        )

        but4 = Button(
            text="Run the dnn",
            font_size="30sp",
            size_hint=(0.8, 0.9),
            pos_hint={"center_x": 0.5},
        )

        but5 = Button(
            text="Exit",
            font_size="30sp",
            size_hint=(0.8, 0.9),
            pos_hint={"center_x": 0.5},
        )
        but5.bind(on_press=self.but5Call)

        rightSideGrid.add_widget(but1)
        rightSideGrid.add_widget(but2)
        rightSideGrid.add_widget(but3)
        rightSideGrid.add_widget(but4)
        rightSideGrid.add_widget(but5)

        # background init
        backgroundGrid.add_widget(leftSideGrid)
        backgroundGrid.add_widget(rightSideGrid)
        self.add_widget(backgroundGrid)

    def but1Call(self, instance):
        print("Button is pressed")
        print("The button % s state is <%s>" % (instance, instance.state))
        self.manager.current = "second"
        self.manager.transition.direction = "left"

    def but2Call(self, instance):
        print("Button is pressed")
        print("The button % s state is <%s>" % (instance, instance.state))
        self.manager.current = "third"
        self.manager.transition.direction = "left"

    def but5Call(self, instance):
        exit()


# diplay img page
class SecondWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # the set up for background area
        firstArea = BoxLayout(orientation="vertical")

        # first area bar (toolbar)
        toolBar = BoxLayout(orientation="horizontal")
        toolBar.size_hint = (1, None)
        toolBar.height = "60dp"

        # back button followed by it's binding to the callback
        backButton = Button(
            size_hint=(None, None),
            pos_hint={"x": 0, "center_y": 0.5},
            width="100dp",
            height="40dp",
            text="go back",
        )

        backButton.bind(on_press=self.callback)
        #########################################################

        # display area
        scrollTry = ScrollView(pos_hint={"center_x": 0.5, "center_y": 0.5}, height=5000)
        print(self.size)
        print("self.size")
        theImagesGrid = StackLayout()

        """FGnet_path = "./datasets/FGNET/newImages/"
        images_path = os.listdir(FGnet_path)
        sizz = 0
        for i in enumerate(images_path):
            sizz = sizz + 1
        """
        dArray = self.imgsAsDArray()

        hieghtScroll = math.ceil(len(dArray) / 4) * 200
        grid = GridLayout(
            size_hint_y=None,
            cols=4,
            row_default_height="200dp",
            row_force_default=True,
            # col_default_width="200dp",
            spacing=(0, 0),
            padding=(0, 0),
            height=str(hieghtScroll) + "dp",
        )

        FGnet_path = "./datasets/FGNET/newImages/"
        for data in dArray:
            path = dArray[data][len(dArray[data]) - 1][1]
            buttIMG = Button(
                background_normal=(FGnet_path + path),
                pos_hint={"center_x": 0.5, "center_y": 0.5},
                size_hint_x=None,
                width="200dp",
            )

            buttIMG.bind(
                on_press=lambda *args, imgArr=dArray[data]: self.dispAll(imgArr)
            )
            # img = Image(source=(FGnet_path + path))  # , size_hint=(0.2, 0.2)
            imgLabel = Label(text=str(data), size_hint=(1, 0.2))

            boxComp = BoxLayout(orientation="vertical")
            boxComp.add_widget(buttIMG)
            boxComp.add_widget(imgLabel)
            grid.add_widget(boxComp)

        """
        for i, path in enumerate(images_path):
            img = Image(source=(FGnet_path + path))  # , size_hint=(0.2, 0.2)
            imgLabel = Label(text=str(i), size_hint=(1, 0.2))
            boxComp = BoxLayout(orientation="vertical")
            boxComp.add_widget(img)
            boxComp.add_widget(imgLabel)
            grid.add_widget(boxComp)
            # if i == 100:
            #   break
        """

        """for i in range(5):
            x = Label(text=str(i))
            theImagesGrid.add_widget(x)"""

        """b1 = Button(text="b1")
        b2 = Button(text="b1")
        b3 = Button(text="b1")
        theImagesGrid.add_widget(b1)
        theImagesGrid.add_widget(b2)
        theImagesGrid.add_widget(b3)"""
        # adding  all elments to the background
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

    # generate the images in a dictionaries
    def imgsAsDArray(self):
        FGnet_path = "./datasets/FGNET/newImages/"
        images_path = os.listdir(FGnet_path)
        finalData = []

        for i, path in enumerate(images_path):
            names = path.split(".")[0]
            names = names.split("A")
            names[1] = names[1].replace("a", "")
            names[1] = names[1].replace("b", "")
            # finalData.append([int(names[0]), int(names[1])])
            finalData.append([names[0], [int(names[1]), path]])

        # print(finalData)

        collectingData = {}
        # collectingData["sd"] = []

        for data in finalData:
            try:
                collectingData[data[0]].append(data[1])
            except:
                # print("An exception occurred")
                collectingData[data[0]] = [data[1]]
        return collectingData

    def dispAll(self, xpic):
        os.chdir(currDirectory)
        try:
            images = []
            FGnet_path = "./datasets/FGNET/newImages/"
            for data in xpic:
                images.append(cv2.imread(FGnet_path + data[1]))
            fig = plt.figure(figsize=(10, 7))
            columns = 4
            rows = math.ceil(len(xpic) / 4)

            for idx, y in enumerate(images):
                fig.add_subplot(rows, columns, idx + 1)
                plt.imshow(y)
                plt.axis("off")
                plt.title("First")
            plt.show()
        except:
            print(xpic)


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

        # bottom bar
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
                on_press=lambda y: self.disp(y=idx)
            )  # on_press=lambda x: self.on_press(x=sum)
            self.leftSide.add_widget(tempButton)

        scrollTry.add_widget(self.leftSide)

        self.rightSide = BoxLayout(orientation="vertical", size_hint=(0.2, 1))

        confirm = Button(
            text="Save to Dataset",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint=(0.9, 1),
        )
        confirm.bind(on_press=self.saveAll)
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

        imageSwapButtons = BoxLayout(
            orientation="horizontal",
            size_hint=(None, None),
            pos_hint={"center_x": 0.6, "center_y": 0.5},
            width="500dp",
            height="40dp",
        )

        orginalImage = Button(
            # size_hint=(None, None),
            # pos_hint={"x": 0, "center_y": 0.5},
            # width="100dp",
            # height="40dp",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint=(0.5, 1),
            text="Original",
        )
        orginalImage.bind(on_press=self.dispOG)

        imageAfterAnotation = Button(
            # size_hint=(None, None),
            # pos_hint={"x": 0, "center_y": 0.5},
            # width="100dp",
            # height="40dp",
            text="After Anottation",
        )
        imageAfterAnotation.bind(on_press=self.dispAI)

        inageAfterPP = Button(
            # size_hint=(None, None),
            # pos_hint={"x": 0, "center_y": 0.5},
            # width="100dp",
            # height="40dp",
            text="Preprocessed",
        )
        inageAfterPP.bind(on_press=self.dispPP)

        imageSwapButtons.add_widget(orginalImage)
        imageSwapButtons.add_widget(imageAfterAnotation)
        imageSwapButtons.add_widget(inageAfterPP)

        toolBar.add_widget(backButton)
        toolBar.add_widget(imageSwapButtons)
        box.add_widget(toolBar)
        box.add_widget(self.midBox)
        box.add_widget(self.bottomBar)
        # box.add_widget(uploadB)  #####################
        self.add_widget(box)

    def upload(self, instance):
        # currDirectory = os.getcwd()
        path = filechooser.open_file(title="Pick a img file..")
        if not path:
            print("{}\nNO IMAGE CHOSEN!!!!\n{}".format("-" * 10, "-" * 10))
            return False

        image_name = re.findall("\w+\.\w+", path[0])
        image_name = image_name[0]
        csvPath = re.sub(image_name[-4:], ".csv", path[0])
        print(path, "\n", csvPath)

        values = addImgClass(path, csvPath)

        listOfValues = [
            values.originalImagePath,
            values.annotImage,
            values.imgAfterPP,
            values.csvFilePath,
        ]
        print(listOfValues)
        self.imgsToSave.append(listOfValues)
        # os.chdir(currDirectory)
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
            tempButton.bind(on_press=lambda *args, hmm=idx: self.disp(hmm))
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

    def dispAI(self, instance):
        if not self.selectedImg is None:
            self.removeMidBox()
            for idx, val in enumerate(self.imgsToSave):
                if idx == self.selectedImg:
                    # pil_img = Image.open(val[0][0])
                    # print(val[1][0])
                    # print(type(pil_img))
                    pil_img = PILImage.fromarray(val[1])

                    print(pil_img)
                    img_bytes = BytesIO()

                    pil_img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    img = Core_Image(BytesIO(img_bytes.read()), ext="png")

                    self.beeld = kiImage()
                    self.beeld.texture = img.texture
                    self.midBox.add_widget(self.beeld)

                    """if os.path.exists("./tempImages") == False:
                        os.makedirs("./tempImages")
                        print("i am herrrreeeeeeeee")

                    im1 = pil_img.save("./tempImages/" + str(idx) + "-1.png")
                    placeHolderForimg = Image(
                        source="./tempImages/" + str(idx) + "-1.png"
                    )
                    self.midBox.add_widget(placeHolderForimg)"""

    def dispOG(self, instance):
        if not self.selectedImg is None:
            self.removeMidBox()
            for idx, val in enumerate(self.imgsToSave):
                if idx == self.selectedImg:
                    placeHolderForimg = Image(source=val[0][0])
                    self.midBox.add_widget(placeHolderForimg)

    def dispPP(self, instance):
        if not self.selectedImg is None:
            self.removeMidBox()
            for idx, val in enumerate(self.imgsToSave):
                if idx == self.selectedImg:
                    pil_img = val[2]
                    img_bytes = BytesIO()
                    pil_img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    img = Core_Image(BytesIO(img_bytes.read()), ext="png")

                    self.beeld = kiImage()
                    self.beeld.texture = img.texture
                    self.midBox.add_widget(self.beeld)

    def saveAll(self, instance):
        # print(self.imgsToSave[0][3]) cvs information
        # df = pd.DataFrame(arr)
        savePath = "./datasets/FGNET/newImages/"
        newLabel = self.maxLabel()
        print(newLabel)
        didItGetSaved = False
        for idx, dat in enumerate(self.imgsToSave):
            im1 = dat[2].save(
                savePath + "0" + str(newLabel) + "A" + str(idx + 1) + ".JPG"
            )
            didItGetSaved = True
            # print(type(dat[2]))
        if didItGetSaved == True:
            self.removeMidBox()
            placeHolderForimg = Label(text="placeHolderForimg")
            self.midBox.add_widget(placeHolderForimg)
            self.removeBeforeUpdate()
            # create content and add to the popup
            content = BoxLayout(orientation="vertical")
            textP = Label(text="Images has been saved with label: 0" + str(newLabel))
            conf = Button(text="Confirm")

            content.add_widget(textP)
            content.add_widget(conf)
            popup = Popup(
                title="Notice!!",
                content=content,
                auto_dismiss=False,
                size_hint=(None, None),
                size=(400, 200),
            )

            # bind the on_press event of the button to the dismiss function
            conf.bind(on_press=popup.dismiss)

            # open the popup
            popup.open()
            didItGetSaved = False

    def maxLabel(self):
        # pp = "D:/Files/Documents D/GitHub/AIFR-with-feature-extraction/datasets/FGNET/newImages"
        os.chdir(currDirectory)  # reset the directory
        FGnet_path = "./datasets/FGNET/newImages"
        images_path = os.listdir(FGnet_path)

        max = 1
        for i, path in enumerate(images_path):
            names = path.split(".")[0]
            names = names.split("A")
            # names[1] = names[1].replace("a", "")
            # names[1] = names[1].replace("b", "")

            if int(names[0]) > max:
                max = int(names[0])

        return max + 1


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("guiCode.kv")


class myTryApp(App):
    def build(self):

        # Window.clearcolor = (1, 1, 1, 1)
        return kv


if __name__ == "__main__":

    myTryApp().run()
