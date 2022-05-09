#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:56:31 2022

@author: jared
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
'''
####################################################################
path_to_audio: is path to mp3 files

files_csv_path: CSV containing mp3 file name numbers

####################################################################
'''
#files = np.genfromtxt("/home/jared/Videos/project3/train.csv", delimiter = ',',dtype = str)
test_files =  np.genfromtxt("/home/jared/Videos/project3/test_idx.csv", delimiter = ',',dtype = str)
Xs1 = []
Xs2 = []
SRs = []
timeseries = []
Sings = []
Sings2 = []
start, end = 0, 27
"""
Compute the MFCC of .mp3 file, then take SVD and create matrix of 21 largest singular values
Mx21 where M is the number of audio files processed.
"""
def computeMFCC(Sings,path_to_audio,limit,files_csv_path):
    files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)
    for i in range(1,limit):
        if (len(files[0]) ==2):
            filename = path_to_audio+ files[i,0]+".mp3"
        else:
            filename = path_to_audio+ files[i]+".mp3"
        x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        mfcc = skl.preprocessing.StandardScaler().fit_transform(x.reshape(1, -1))
        #librosa.display.specshow(mfcc, sr=sr, x_axis='time');
        #COMP SVD OF MFCC. COMMENT OUT FOR just MFCCS
        #Better results with 25
        X = skl.decomposition.TruncatedSVD(n_components = 21).fit(mel)
        A=X.singular_values_.tolist()
        B=list(A)
        if (len(files[0]) ==2):
            B.insert(len(A),int(files[i][1]))
        Sings.append(B)
    return Sings

"""
Compute MFCC in a way that is friendly to parallel processing call to increase 
speed
"""
def computeMFCC_Parallel(i,path_to_audio,files_csv_path):
        files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)

        if (len(files[0]) ==2):
            filename = path_to_audio+ files[i,0]+".mp3"
        else:
            filename = path_to_audio+ files[i]+".mp3"
        x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        return mfcc
"""
Create and save spectogram of audio file. Written to be parallel friendly
"""
def computeSpec(i,Xs,Path_to_audio_files,files_csv_path):
    files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)

    print(files[i])
    filename = Path_to_audio_files + files[i][0]+".mp3"
        #All songs seem to be at least 29s long
    x, sr = librosa.load(filename, sr=None, mono=True,duration=29)

    plt.figure()

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)

    librosa.display.specshow(log_mel,cmap = 'coolwarm', sr=sr, hop_length=512, x_axis='time')
    
    plt.savefig('Spectograms' +str(i)+'.png')
    plt.show()


if __name__ == "__main__":


    p_to_audio = input("Path to audio files: \n")
    p_to_csv = input("Path to test or training csv: \n")
    #Only run the Parallel job once. Thses will consume a lot of memory and aren't necessary to run unles you make chages

    #Parallel(n_jobs = -1)(delayed(computeSpec)(i,Xs1,p_to_audio,p_to_csv) for i in range(1,2401))
    computeMFCC(Sings,p_to_audio,2401,p_to_csv)

#PARALLEL CALL TO SPEED UP IF NEEDING MFCC AS INPUT
    #Parallel(n_jobs = -1)(delayed(computeMFCC_Parallel)(i,p_to_audio) for i in range(1,1201))
    Sings = np.array(Sings)
    newlist = [Sings[x][-1] for x in range(len(Sings))]
    B = np.delete(Sings, -1, axis=1)
    print(newlist)
    skl.preprocessing.normalize(B, norm='l2')
    
    x_train, x_test, y_train, y_test = train_test_split(B, newlist, test_size=0.2, shuffle=True)
    #Try w/150ish iter to improve acc
    logReg = skl.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter = 119)
    logReg.fit(x_train,y_train)
    #Score model accuracy for validation data
    score = logReg.score(x_test, y_test)
    y_pred = logReg.predict(x_test)
#Conf. matrix
    cf= sklearn.metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cf,annot=True)
    print(score)
        #If yes, run prediction on test data
    while True:
        text = input("Continue to test? (y/n)\n")
        if text == 'y':
            p_to_audio2 = input("Path to audio files: \n")
            p_to_csv2 = input("Path to test or training csv: \n")
            computeMFCC(Sings2,p_to_audio2,1201,p_to_csv2)
            Sings2 = np.array(Sings2)
            skl.preprocessing.normalize(Sings2, norm='l2')
            predictions = logReg.predict(Sings2)
            print(predictions)
            break
        elif text == 'n':
            break

#


