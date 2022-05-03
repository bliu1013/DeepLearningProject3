#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:53:49 2022

@author: jared
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
#from keras.preprocessing import image
#from tensorflow.keras.models import load_model
#import sklearn
#from sklearn import preprocessing
#import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io

import csv

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


import librosa
import imageio



# Read in spectrogram data from pngs
def get_data():
    x = []
    x_val = []
    y = []
    y_val = []

    #Shuffle the order of data
    specs = np.linspace(1, 2400, 2400, dtype=int)
    random.shuffle(specs)
    for num, i in enumerate(specs):
        print("Reading image", (num))
        file = f"Spectrograms/{i}.png"
        img = io.imread(file, as_gray=True)

        #Get rid of extra pixels
        img = img[60:426,81:575]
        print(img)
        if i <= 401:
            class_num = 0
        elif i <= 801:
            class_num = 1
        elif i <= 1201:
            class_num = 2
        elif i <= 1601:
            class_num = 3
        elif i <= 2001:
            class_num = 4
        else:
            class_num = 5    

        #Split data for validation
        if i % 5 == 0:
            x_val.append(img)
            y_val.append(class_num)
        else:
            x.append(img)
            y.append(class_num)

    return x, x_val, y, y_val

  
 
 
def make(input_shapes):
    inputs = keras.Input(shape=input_shapes)
    x = layers.Rescaling(1./255, offset=0.0)(inputs)   
    x = layers.Conv2D(32,3,strides=2,padding = "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64,3,strides=2,padding = "same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(.2)(x)
    outputs = layers.Dense(6,activation = "softmax")(x)
    return keras.model(inputs,outputs)

    


            

if __name__ == "__main__":
    ######################
    # x_data, x_valid, y_data, y_valid = get_data()
    # #print(np.shape(x_data), y_data)
    # x_data = np.expand_dims(x_data, axis=-1)
    # x_valid = np.expand_dims(x_valid, axis=-1)
    # model = train(x_data, x_valid, y_data, y_valid)

    # while True:
    #     text = input("Continue to test? (y/n)\n")

    #     if text == 'y':
    #         test(model)
    #         break
    #     elif text == 'n':
    #         break
###################
    train_datagen = ImageDataGenerator(rescale = 1/255.0)
    valid_datagen = ImageDataGenerator(rescale = 1/255.0)
    
    train_ds = train_datagen.flow_from_directory(
    directory="Data/training",
    target_size = (256,256),
    batch_size=16,
    class_mode="categorical"
    )

    valid_ds = valid_datagen.flow_from_directory(
    directory="Data/validation",
    target_size = (256,256),
    batch_size=16,
    class_mode="categorical"
    )
    

    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding="same", input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3),padding="same", activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3),padding="same", activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3),padding="same", activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    model.summary()
    #model = make((400,400))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])   
    
    model.summary()
    
    history = model.fit(train_ds, epochs=30, validation_data=valid_ds)    
    
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
