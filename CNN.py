#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:53:49 2022

@author: jared
"""
import sys

import sklearn
from sklearn import preprocessing
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
import random

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


def train(x_data, x_valid, y_data, y_valid):
    #Create model

    model = keras.models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(366, 494, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))



    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])  

    #Convert all lists to nparrays
    x_data = np.asarray(x_data)
    x_valid = np.asarray(x_valid)
    y_data = np.asarray(y_data)
    y_valid = np.asarray(y_valid)

    #Run
    history = model.fit(x_data, y_data, epochs=300, validation_data=(x_valid, y_valid))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()

    return model


def test(model):
    #Get test example indices
    indices = []
    with open('test_idx.csv') as file:
        reader = csv.reader(file)

        for idx in reader:
            if idx != ['new_id']:
                indices.append(idx)

    x = []
    for i in range(1, 1201):
        print("Testing image", (i))
        file = f"test_spec/{i}.png"
        img = io.imread(file, as_gray=True)
        img = img[60:426,81:575]
        x.append(img)
    x = np.expand_dims(x, axis=-1)

    x = np.asarray(x)
    y_pred = np.argmax(model.predict(x), axis=-1)

    with open('solution_CNN.csv', 'w+', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['id' , 'genre'])
        for i in range(1200):
            writer.writerow([indices[i][0], y_pred[i]])


if __name__ == "__main__":
    x_data, x_valid, y_data, y_valid = get_data()
    #print(np.shape(x_data), y_data)
    x_data = np.expand_dims(x_data, axis=-1)
    x_valid = np.expand_dims(x_valid, axis=-1)
    model = train(x_data, x_valid, y_data, y_valid)

    while True:
        text = input("Continue to test? (y/n)\n")

        if text == 'y':
            test(model)
            break
        elif text == 'n':
            break