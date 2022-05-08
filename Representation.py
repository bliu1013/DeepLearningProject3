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



files = np.genfromtxt("/home/jared/Videos/project3/train.csv", delimiter = ',',dtype = str)
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
def computeMFCC(Sings,path_to_audio,files,limit,files_csv_path):
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


def computeMFCC_Parallel(i,path_to_audio,files,files_csv_path):
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

def computeSpec(i,Xs,Path_to_audio_files,files_csv_path):
    files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)

    print(files[i])
    filename = Path_to_audio_files + files[i][0]+".mp3"
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

    librosa.display.specshow(log_mel,cmap = 'coolwarm', sr=sr, hop_length=512, x_axis='time')
    
    plt.savefig('Spectograms' +str(i)+'.png')
    plt.show()


def compSVD_dft(Xs,Path_to_audio_files,files_csv_path):
    files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)

    for i in range(1,2401):
        if(len(files[i,0])==6):
            filename = Path_to_audio_files + files[i,0]+".mp3"
        else:
            filename = Path_to_audio_files +files[i,0]+".mp3"
        x, sr = librosa.load(filename, sr=None, mono=True,duration=29)
        #Xs.append(x.tolist())
        print(x.shape)
        #dft = np.abs(librosa.stft(x))
        #dft = librosa.amplitude_to_db(dft, ref=np.max)
        #print(dft.shape)
        #plt.plot(dft)
        svd = skl.decomposition.TruncatedSVD(n_components = 50).fit(x.reshape(1,-1))
        print("SVD",svd.singular_values_.T)
        print(svd.singular_values_.T.flatten())
        Xs.append(svd.singular_values_)
    return Xs

def comp_for_SVD(Xs, genre,Path_to_audio_files,files_csv_path):
    files = np.genfromtxt(files_csv_path, delimiter = ',',dtype = str)

    for i in range(1,401):
        #if(files[i][1]==str(genre)):
            print(files[i][0])
            
            filename = Path_to_audio_files + files[i,0]+".mp3"
            x, sr = librosa.load(filename, sr=None, mono=True,duration=15)
            Xs.append(x.tolist())
            print(len(Xs))
    svd = skl.decomposition.TruncatedSVD(n_components = 5).fit(Xs)
    print("SVD",svd.singular_values_)
    return Xs

#Only run the Parallel job once.
#Parallel(n_jobs = -1)(delayed(computeSpec)(i,Xs1) for i in range(1,2401))
#computeSpec(4, Xs1)
computeMFCC(Sings,"/home/jared/Downloads/project3/train/",files,2401,"/home/jared/Videos/project3/train.csv")

#PARALLEL CALL TO SPEED UP IF NEEDING MFCC AS INPUT
#Parallel(n_jobs = -1)(delayed(computeMFCC_Parallel)(i,"/home/jared/Downloads/project3/train/") for i in range(1,1201))

Sings = np.array(Sings)
newlist = [Sings[x][-1] for x in range(len(Sings))]
B = np.delete(Sings, -1, axis=1)
print(newlist)
skl.preprocessing.normalize(B, norm='l2')

x_train, x_test, y_train, y_test = train_test_split(B, newlist, test_size=0.2, shuffle=True)
#Try w/150ish iter to improve acc
logReg = skl.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter = 119)
logReg.fit(x_train,y_train)

score = logReg.score(x_test, y_test)
y_pred = logReg.predict(x_test)
cf= sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cf)
print(score)

while True:
        text = input("Continue to test? (y/n)\n")
        if text == 'y':
            computeMFCC(Sings2,"/home/jared/Downloads/project3/test/",test_files,1201)
            Sings2 = np.array(Sings2)
            skl.preprocessing.normalize(Sings2, norm='l2')
            predictions = logReg.predict(Sings2)
            print(predictions)
            break
        elif text == 'n':
            break

#


