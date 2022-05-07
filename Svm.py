import sys
import csv
import tensorflow
import imageio
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import csv
from sklearn.datasets import make_blobs


from tensorflow import keras
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read in spectrogram data from pngs
def get_data_validation():
    x = []
    x_val = []
    y = []
    y_val = []
    df=pd.read_csv('train2.csv')
    classifier = df['genre'].tolist()

    #Shuffle the order of data
    specs = np.linspace(1, 2400, 2400, dtype=int)
    random.shuffle(specs)
    for num, i in enumerate(specs):
        print("Reading image", (num))
        file = f"Spectrograms/{i}.png"
        img = io.imread(file, as_gray=True)

        #Get rid of extra pixels
        img = img[60:426,81:575]
        img = img.flatten()
        #Split data for validation
        #if i % 5 == 0:
        #    x_val.append(img)
        #    y_val.append(classifier[i-1])
        #else:
        #    x.append(img)
        #    y.append(classifier[i-1])
        x.append(img)
        y.append(classifier[i-1])
    #x and y are training data x_val, y_val are validation datasets
    return x, x_val, y, y_val

def svm_train(x,y):
    clf = svm.SVC(kernel='poly', degree=2, C=2)
    return clf.fit(x,y)

def get_data_test():
    x = []
    y = []
    df=pd.read_csv('test_idx.csv')
    identifiers = df['new_id'].tolist()

    #Shuffle the order of data
    specs = np.linspace(1, 1200, 1200, dtype=int)
    #random.shuffle(specs)
    for num, i in enumerate(specs):
        print("Reading image", (num))
        file = f"test_spec/{i}.png"
        img = io.imread(file, as_gray=True)

        #Get rid of extra pixels
        img = img[60:426,81:575]
        img = img.flatten()

        x.append(img)
    return x, identifiers

def classifier(ids, classes):
    file_object = open('classified.csv', 'w+')
    file_object.write("id,genre\n")
    k = 0
    for id in ids:
        file_object.write(str(id) + "," + str(classes[k]) + '\n')
        k= k+1

def validation_testing(y_predict, y_actual):
    """
    80/20 validation testing
    :param y_predict: list of predicted classes
    :param y_actual: list of actual classes
    :return: int accuracy
    """
    k = 0
    right = 0
    total = len(y_actual)
    for item in y_predict:
        if item == y_actual[k]:
            right = right + 1
            k = k+1
        else:
            k = k+1
    accuracy = right/total
    print(accuracy)


if __name__ == "__main__":
    #Code to begin validation
    x, x_val, y, y_val = get_data_validation()
    for i in range(5):
        fifth = len(x)/5
        x_val = x[i*fifth:(i+1)*fifth]
        y_val = y[i*fifth:(i+1)*fifth]
        x_train = x[:i*fifth] + x[(i+1)*fifth:]
        y_train = x[:i*fifth] + x[(i+1)*fifth:]
        clf = svm_train()
    #clf = svm_train(x, y)
    #first_class = clf.predict(x_val)
    #Function to begin valication accuracy
    #validation_testing(first_class, y_val)
    #Code to begin classification of testing dataset
    x, x_val, y, y_val = get_data_validation()
    x_val, y_val = get_data_test()
    clf = svm_train(x, y)
    first_class = clf.predict(x_val)
    classifier(y_val, first_class)


