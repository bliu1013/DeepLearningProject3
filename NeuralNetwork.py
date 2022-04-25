import sys

import sklearn

import tensorflow

from tensorflow import keras

from skimage import io


import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


import librosa
import imageio



# Read in spectrogram data from pngs

def get_data():
    x = []
    validation = []

    for i in range(2400):
        print("Reading image", (i+1))
        file = f"Spectrograms/{i+1}.png"
        img = io.imread(file, as_gray=True)

        #Get rid of extra pixels
        img = img[60:426,81:575]

        if i % 2 == 0:
            x.append(img)
        else:
            validation.append(img)
    
    #Manually create array of classes
    y = []
    y.extend(np.zeros(200))
    y.extend(np.ones(200))
    y.extend((np.ones(200) * 2))
    y.extend((np.ones(200) * 3))
    y.extend((np.ones(200) * 4))
    y.extend((np.ones(200) * 5))

    return x, validation, y, y


def train(x_data, x_valid, y_data, y_valid):
    #Create model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[366, 494]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

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
    history = model.fit(x_data, y_data, epochs=50, validation_data=(x_valid, y_valid))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    x_data, x_valid, y_data, y_valid = get_data()
    #print(np.shape(x_data), y_data)

    train(x_data, x_valid, y_data, y_valid)
