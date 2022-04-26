import sys
import csv

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
    x_val = []
    y = []
    y_val = []

    class_num = 0
    for i in range(1, 2401):
        print("Reading image", (i))
        file = f"Spectrograms/{i}.png"
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
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dense(300, activation="relu"))
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
    history = model.fit(x_data, y_data, epochs=5, validation_data=(x_valid, y_valid))
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
            if idx != 'new_id':
                indices.append(idx)

    x = []
    for i in range(1, 1201):
        print("Testing image", (i))
        file = f"test_spec/{i}.png"
        img = io.imread(file, as_gray=True)
        img = img[60:426,81:575]
        x.append(img)
    
    x = np.asarray(x)
    y_pred = np.argmax(model.predict(x), axis=-1)

    #print(y_pred)

    with open('solution.csv', 'w+') as out:
        writer = csv.writer(out)
        writer.writerow('id,genre')
        for i in range(1200):
            writer.writerow(f'[{indices[i]}],[{y_pred[i]}]')


if __name__ == "__main__":
    x_data, x_valid, y_data, y_valid = get_data()

    model = train(x_data, x_valid, y_data, y_valid)

    #Prompt to continue to testing
    if input("Continue to test? (y/n)\n") == 'y':
        test(model)

    
