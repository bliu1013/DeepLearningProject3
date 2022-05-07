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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read in spectrogram data from pngs
def get_data_validation():
    """
    Function to get the training data used to train the SVM
    :return: x values that hold the values and y hold the classifications
    """
    x = []
    x_val = []
    y = []
    y_val = []
    #read from the training data
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

def svm_train(x, y):
    """
    Function used to train the SVM
    :param x: x values used for training
    :param y: classes used for training
    :return: The trained model
    """
    clf = svm.SVC(kernel='rbf', C=2)
    return clf.fit(x, y)

def get_data_test():
    """
    Function used to create the lists for the testing data
    :return: x values for predicting and the identification numbers.
    """
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
    """
    Function to generate file of the classification of the testing data
    :param ids: id's taken from the training.csv file
    :param classes: classified
    :return: None
    """
    file_object = open('classified.csv', 'w+')
    file_object.write("id,genre\n")
    k = 0
    for id in ids:
        file_object.write(str(id) + "," + str(classes[k]) + '\n')
        k= k+1

#Taken from GeekstoGeeks
def average(lst):
    return sum(lst)/len(lst)

def validation_testing(y_predict, y_actual):
    """
    validation testing
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
    return accuracy


if __name__ == "__main__":
    #Code to begin 5-point cross validation
    x, x_val, y, y_val = get_data_validation()
    accuracies = []
    for i in range(5):
        fifth = len(x)//5
        x_val = x[i*fifth:(i+1)*fifth]
        y_val = y[i*fifth:(i+1)*fifth]
        x_train = x[:i*fifth] + x[(i+1)*fifth:]
        y_train = y[:i*fifth] + y[(i+1)*fifth:]
        clf = svm_train(x_train, y_train)
        classes = clf.predict(x_val)
        print("calculating accuracies for "+str(i+1))
        accuracies.append(validation_testing(classes, y_val))
        if i == 1:
            spec_array = classes
            actual = y_val
            predicted = spec_array
            disp_label = [0,1, 2, 3, 4, 5]
            matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2, 3, 4, 5])
            print(matrix)
            disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=disp_label)
            disp.plot()
            plt.show()

    print(average(accuracies))
    #clf = svm_train(x, y)
    #first_class = clf.predict(x_val)
    #Function to begin valication accuracy
    #validation_testing(first_class, y_val)
    #Code to begin classification of testing dataset
    #x, x_val, y, y_val = get_data_validation()
    #x_val, y_val = get_data_test()
    #clf = svm_train(x, y)
    #first_class = clf.predict(x_val)
    #classifier(y_val, first_class)

    # spec_array = classify_conf(new_class_matrix, p_v, prob_calcs)
    # actual = cols_comp
    # predicted = spec_array
    # disp_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # matrix = confusion_matrix(actual,predicted, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20])
    # print(matrix)
    # disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=disp_label)
    # disp.plot()
    # plt.show()