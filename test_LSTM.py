import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
import glob
import cv2
import keras
import csv
import pickle

from keras.layers import Input, Dense,Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model

from sklearn.model_selection import train_test_split

def getdata():
    with open('test_features', 'rb') as fp:
        X=pickle.load(fp)

    with open('test_ids', 'rb') as fp:
        ids=pickle.load(fp)

    return np.array(X), np.array(ids)


X,ids=getdata()

X=np.reshape(X,(len(X),15,256))

model=load_model('LSTM_model.h5')
X_pred=model.predict(X)

arr=[]

i=0
for x in X_pred:
    arr.append([ids[i],x[0],x[1],x[2],x[3],x[4],x[5]])
    i+=1

csvfile = "mysubmission.csv"

i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["Id","toxic","severe_toxic","obscene","threat"
        ,"insult","identity_hate"])
    i+=1
    writer.writerows(arr)

print("data dumped to file")