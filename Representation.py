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


file_directory = "/home/jared/Videos/project3/train";

files = np.genfromtxt("/home/jared/Videos/project3/train.csv", delimiter = ',',dtype = str)
test_files =  np.genfromtxt("/home/jared/Videos/project3/test_idx.csv", delimiter = ',',dtype = str)
Xs1 = []
Xs2 = []
SRs = []
timeseries = []
Sings = []
start, end = 0, 27
#ipd.Audio(data=x[start*sr:end*sr], rate=sr)

def computeMFCC(Sings,path_to_audio):
    for i in range(390,410):
        filename = path_to_audio+ files[i,0]+".mp3"
       
        x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))
        mfcc = skl.preprocessing.StandardScaler().fit_transform(x.reshape(1, -1))
        #librosa.display.specshow(mfcc, sr=sr, x_axis='time');
        #COMP SVD OF MFCC. COMMENT OUT FOR just MFCCS
        X = skl.decomposition.TruncatedSVD(n_components = 100).fit(mel)
        A=X.singular_values_.tolist()
        B=list(A)
        B.insert(len(A),int(files[i][1]))
        Sings.append(B)
    return Sings


def computeSpec(i,Xs):
    print(files[i])
    filename = "/home/jared/Downloads/project3/train/" + files[i][0]+".mp3"
        #All songs seem to be at least 29s long
    x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
    #librosa.display.waveplot(Xs[i], SRs[i], alpha=0.5);
    #plt.vlines([start, end], -1, 1)
    #start = len(x) // 2
    plt.figure()
    # plt.plot(Xs[i][start:start+2000])
    # plt.ylim((-1, 1));
    # plt.show()
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)

    librosa.display.specshow(log_mel,cmap = 'gray', sr=sr, hop_length=512, x_axis='time')
    
    #plt.savefig('/home/jared/CS429/test_spec/' +str(i)+'.png')
    plt.show()


def compSVD_dft(Xs):
    for i in range(1,2401):
        if(len(files[i,0])==6):
            filename = "/home/jared/Downloads/project3/train/" + files[i,0]+".mp3"
        else:
            filename = "/home/jared/Downloads/project3/train/" +files[i,0]+".mp3"
        x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
        #Xs.append(x.tolist())
        print(x.shape)
        #dft = np.abs(librosa.stft(x))
        #dft = librosa.amplitude_to_db(dft, ref=np.max)
        #print(dft.shape)
        #plt.plot(dft)
        svd = skl.decomposition.TruncatedSVD(n_components = 100).fit(x.reshape(1,-1))
        print("SVD",svd.singular_values_.T)
        print(svd.singular_values_.T.flatten())
        Xs.append(svd.singular_values_)
    return Xs

def comp_for_SVD(Xs, genre):
    for i in range(1,401):
        #if(files[i][1]==str(genre)):
            print(files[i][0])
            
            filename = "/home/jared/Videos/project3/train/" + files[i,0]+".mp3"
            x, sr = librosa.load(filename, sr=None, mono=True,duration=15)
            Xs.append(x.tolist())
            print(len(Xs))
    svd = skl.decomposition.TruncatedSVD(n_components = 5).fit(Xs)
    print("SVD",svd.singular_values_)
    return Xs

#Only run the Parallel job once.
#Parallel(n_jobs = -1)(delayed(computeSpec)(i,Xs1) for i in range(1,1201))
#computeSpec(4, Xs1)
computeMFCC(Sings,"/home/jared/Downloads/project3/train/")
Sings = np.array(Sings)
print(Sings[:,-1])
x_train, x_test, y_train, y_test = train_test_split(Sings[:,-1], Sings[:,-1], test_size=0.25, random_state=0)

#skl.preprocessing.normalize(Sings[:,:-1], norm='l2')
logReg = skl.linear_model.LogisticRegression()
logReg.fit(x_train,y_train)





