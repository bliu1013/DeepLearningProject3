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
    n_files = 2400

    x = []
    x_val = []
    y = []
    y_val = []

    class_num = 0
    for i in range(n_files):
        print("Reading image", (i+1))
        file = f"Spectrograms/{i+1}.png"
        img = io.imread(file, as_gray=True)

        #Get rid of extra pixels
        img = img[60:426,81:575]

        #Split data for validation
        if i % 4 == 0:
            x_val.append(img)
            y_val.append(class_num)
        else:
            x.append(img)
            y.append(class_num)

        if (i+1) % (600) == 0:
            class_num += 1

    return x, x_val, y, y_val


def train(x_data, x_valid, y_data, y_valid):
    #Create model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[366, 494]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(300, activation="relu"))
    #model.add(keras.layers.Dense(30, activation="relu"))
    #model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dense(6, activation="softmax"))

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
    history = model.fit(x_data, y_data, epochs=30, validation_data=(x_valid, y_valid))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()

    return model


def test(model):
    x = []
    n_files = 2400
    for i in range(n_files):
        print("Testing image", (i+1))
        file = f"Spectrograms/{i+1}.png"
        img = io.imread(file, as_gray=True)
        img = img[60:426,81:575]
        x.append(img)
    
    y_pred = np.argmax(model.predict(x), axis=-1)

    print(y_pred)


if __name__ == "__main__":
    x_data, x_valid, y_data, y_valid = get_data()
    #print(np.shape(x_data), y_data)

    model = train(x_data, x_valid, y_data, y_valid)

    test(model)
