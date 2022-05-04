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

    train_datagen = ImageDataGenerator(rescale = 1/255.0)
    valid_datagen = ImageDataGenerator(rescale = 1/255.0)
    
    train_ds = train_datagen.flow_from_directory(
    directory="Data/training",
    target_size = (480,480),
    batch_size=16,
    class_mode="categorical",
    shuffle=(True)
    )

    valid_ds = valid_datagen.flow_from_directory(
    directory="Data/validation",
    target_size = (480,480),
    batch_size=16,
    class_mode="categorical"
    )
    

    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 480, 3)))
    model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3),padding="same", activation='relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3),padding="same", activation='relu'))
    # model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    
    # model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
    # model.add(layers.BatchNormalization())   
    # model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
    # model.add(layers.BatchNormalization())   
    # model.add(layers.Conv2D(256, (3, 3),padding="same", activation='relu'))
    # model.add(layers.BatchNormalization())   
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(.2))
    model.add(layers.Dense(6, activation='softmax'))
    model.summary()
    #model = make((400,400))
    model.compile(loss=tf.keras.losses.KLDivergence(),
                  optimizer=tf.keras.optimizers.SGD()(
    learning_rate=0.0000001),
                  metrics=["accuracy"])   
    
    
    history = model.fit(train_ds, epochs=50, validation_data=valid_ds)    
    
    img = keras.preprocessing.image.load_img(
    "test_spec/1.png", target_size=(480,480)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(score)
   
    img = keras.preprocessing.image.load_img(
    "test_spec/105.png", target_size=(480,480)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(score)
    
    img = keras.preprocessing.image.load_img(
    "test_spec/700.png", target_size=(480,480)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(score)
    
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
