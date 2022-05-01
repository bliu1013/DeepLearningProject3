import sys
import csv
import tensorflow
import imageio
import random

from tensorflow import keras
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    keras.backend.clear_session()
    np.random.seed(42)
    tensorflow.random.set_seed(42)

    #Create model
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[366, 494]))
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(500, activation="relu"))
    model.add(keras.layers.Dense(50, activation="relu"))

    model.add(keras.layers.Dense(6, activation="softmax"))

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=1e-8)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])  

    #Convert all lists to nparrays
    x_data = np.asarray(x_data)
    x_valid = np.asarray(x_valid)
    y_data = np.asarray(y_data)
    y_valid = np.asarray(y_valid)

    #Run
    history = model.fit(x_data, y_data, epochs=100, validation_data=(x_valid, y_valid))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
    plt.savefig('loss_function.pdf')

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
    
    x = np.asarray(x)
    y_pred = np.argmax(model.predict(x), axis=-1)

    with open('solution.csv', 'w+', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['id' , 'genre'])
        for i in range(1200):
            writer.writerow([indices[i][0], y_pred[i]])


if __name__ == "__main__":
    x_data, x_valid, y_data, y_valid = get_data()

    model = train(x_data, x_valid, y_data, y_valid)

    #Prompt to continue to testing
    while True:
        text = input("Continue to test? (y/n)\n")

        if text == 'y':
            test(model)
            break
        elif text == 'n':
            break

    
