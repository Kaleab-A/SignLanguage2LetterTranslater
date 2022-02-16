# A simple tkinter window
import tkinter as tk
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from threading import *

import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from random import randint
import keras

class App:
    def __init__(self):
        self.cnt = 1
        self.root = tk.Tk()
        self.root.geometry("800x815")
        self.root.title("Title Here")

        self.videoCapture = cv2.VideoCapture(2)
        self.canvas = Canvas(self.root, width=600, height=500)
        self.filePath = "vgg3_adam_sparse.hdf5"

        self.model = self.model_vgg3()
        self.model.load_weights(self.filePath) 

        self.GUI()

        self.updateImage()
        self.root.mainloop()

        # vid.release()
        # cv2.destroyAllWindows()

    def GUI(self):
        self.label = ttk.Label(self.root, padding=10, text="Sign Language to Text Translator")
        self.label.config(font=("Comic Sans MS",40))
        self.label.pack()
        self.canvas.pack()

        self.T = Text(self.root, height=10, width = 95)
        self.T.pack()

        self.btn = Button(self.root, text="Start", command=self.predictSign)
        self.btn.pack()

        self.panel = Label(text="hello")
        self.panel.pack()

    def prepare_image(self, frame):
        image = Image.fromarray(frame)
        image = np.array(image.resize([50, 50])).astype(float)
        image /= 255
        return image.reshape((1, 50, 50, 3))

    def model_vgg3(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform", input_shape=(50, 50, 3)))
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform"))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform"))
        model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform"))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(Dropout(0.3))

        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform"))
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_uniform"))
        model.add(MaxPooling2D(pool_size=2))
        # model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(30, activation='softmax'))

        return model

    def predictSign(self):
        self.frame_inp = self.prepare_image(self.frame)
        raw_prediction = self.model.predict(self.frame_inp)
        prediction = np.argmax(raw_prediction)
        # confidence = (raw_prediction[0][prediction] / np.sum(raw_prediction)) * 100
        # print(prediction, confidence, "%")
        return self.addText(chr(prediction + ord("A")))

    def addText(self, x):
        if x == "\\": x=" Nothing "
        elif x == "]": x="Space"
        self.T.insert(END, x)

    def updateImage(self):
        ret, self.frame = self.videoCapture.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image = self.photo, anchor="nw")
            if self.cnt % 75 == 0:
                self.predictSign()
            self.cnt += 1
        
        self.root.after(15, self.updateImage)

app1 = App()








