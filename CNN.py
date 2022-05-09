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
            
if __name__ == "__main__":

    train_datagen = ImageDataGenerator(rescale = 1/255.0,validation_split=0.2)
    #valid_datagen = ImageDataGenerator(rescale = 1/255.0)
#Create training data set from Data directory

    train_ds = train_datagen.flow_from_directory(
    directory="Data/training",
    target_size = (480,480),
    batch_size=32,
    class_mode="categorical",
    subset='training',
    shuffle=(True)
    )
#Create validation data set from Data directory
    valid_ds = train_datagen.flow_from_directory(
    directory="Data/training",
    target_size = (480,480),
    batch_size=32,
    class_mode="categorical",
    subset='validation'
    )

#Model creation starting with relu activation
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 480, 3)))  
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(6, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(
    learning_rate=1e-8),
                  metrics=["accuracy"])   
    
    
    model.summary()
    #fit model
    history = model.fit(train_ds, epochs=30, validation_data=valid_ds)    

    
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
