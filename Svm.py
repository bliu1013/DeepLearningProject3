import random
import matplotlib.pyplot as plt
from sklearn import svm

from skimage import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
from joblib import Parallel,delayed

import numpy as np
import pandas as pd


#Section of code was adapted from https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
def computeMFCC(limit):
    """
    Function to compute ceptral coefficients
    :param limit: int that is the number of files to use
    :return: an array of coefficients, one per file. An array of classes, one per file and identifiers
    """
    df=pd.read_csv('train2.csv')
    identifiers = df['new_id'].tolist()
    classes = df['genre'].tolist()
    #print(classes)
    Sings = []
    sings_validation = []
    class_validation = []
    SingsClasses = []
    length = 43080
    for i in range(limit):
        print("precessing " + str(i))
        filename = f"train/{str(identifiers[i]).zfill(8)}.mp3"
        x, sr = librosa.load(filename, sr=None, mono=True, duration=25)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        mfcc = mfcc.flatten()
        if len(mfcc) > 43080:
            mfcc = mfcc[0:43080]
        if len(mfcc) < 43080:
            print("Warning: mfcc data too small; padding " + str(len(mfcc)))
            mfcc = mfcc + [0.0]*(43080 - len(mfcc))
        assert(len(mfcc) == 43080)
        Sings.append(mfcc)
        SingsClasses.append(classes[i])
        # Split data for validation
        #if i % 5 == 0:
        #    sings_validation.append(mfcc)
        #    class_validation.append(classes[i])
        #else:
        #    Sings.append(mfcc)
        #   SingsClasses.append(classes[i])
    return Sings, SingsClasses, identifiers, class_validation
        #, sings_validation, class_validation

def computeMFCC_test(limit):
    """
    This function computes MFCC coefficients for
    :param limit:
    :return:
    """
    df=pd.read_csv('test_idx.csv')
    identifiers = df['new_id'].tolist()
    Sings = []
    length = 43080
    for i in range(limit):
        print("precessing " + str(i))
        filename = f"test/{str(identifiers[i]).zfill(8)}.mp3"
        x, sr = librosa.load(filename, sr=None, mono=True, duration=25)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        mfcc = mfcc.flatten()
        if len(mfcc) > 43080:
            mfcc = mfcc[0:43080]
        if len(mfcc) < 43080:
            print("Warning: mfcc data too small; padding " + str(len(mfcc)))
            fill = np.array([0.0]*(43080 - len(mfcc)))
            mfcc = np.append(mfcc, fill)
        assert(len(mfcc) == 43080)
        Sings.append(mfcc)
    return Sings, identifiers


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

#Section of code used from the SVM workbook on UNM Learn
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

    #Code to begin training and testing the svm model with MFCC data
    Mfcc, classes, somevalue, othervalue = computeMFCC(2400) #2400
    #Mfcc, ids = computeMFCC(2400)
    clf = svm_train(Mfcc, classes)
    validation_x, identifiers = computeMFCC_test(1200) #1200
    identifiers = identifiers[:1200]
    new_classes = clf.predict(validation_x)
    classifier(identifiers, new_classes)

###############################################################
    #Code to begin 5-point cross validation
    #x, y, x_val, y_val = computeMFCC(2400) #2400
    #x, x_val, y, y_val = get_data_validation()
    #accuracies = []
    #for i in range(5):
    #    fifth = len(x)//5
    #    x_val = x[i*fifth:(i+1)*fifth]
    #    y_val = y[i*fifth:(i+1)*fifth]
    #    x_train = x[:i*fifth] + x[(i+1)*fifth:]
    #    y_train = y[:i*fifth] + y[(i+1)*fifth:]
    #    clf = svm_train(x_train, y_train)
    #    classes = clf.predict(x_val)
    #    print("calculating accuracies for "+str(i+1))
    #    accuracies.append(validation_testing(classes, y_val))
    #    print(accuracies)
    #    if i == 1:
    # Code to print confusion Matrix
    #        spec_array = classes
    #        actual = y_val
    #        predicted = spec_array
    #        disp_label = [0,1, 2, 3, 4, 5]
    #        matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2, 3, 4, 5])
    #        print(matrix)
    #        disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=disp_label)
    #        disp.plot()
    #        plt.show()

#######################################################################
    #Code to begin classification of testing dataset for Spectrogram data
    #x, x_val, y, y_val = get_data_validation()
    #x_val, y_val = get_data_test()
    #clf = svm_train(x, y)
    #first_class = clf.predict(x_val)
    #classifier(y_val, first_class)

