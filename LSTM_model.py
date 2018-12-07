import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
import glob
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import os
import time
import imageio
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2
import keras

import pickle

from keras.layers import Input, Dense,Activation, LSTM, MaxPooling1D, Flatten, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model

from sklearn.model_selection import train_test_split


def getdata():
    with open('train_features', 'rb') as fp:
        X=pickle.load(fp)

    with open('labels', 'rb') as fp:
        y=pickle.load(fp)

    return np.array(X), np.array(y)


X,y=getdata()

print(X.shape)
print(y.shape)

X=np.reshape(X,(len(X),15,256))

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

input_img = Input(shape=(15,256))  # adapt this if using `channels_first` image data format

x=LSTM(64,return_sequences=True)(input_img)
x=MaxPooling1D()(x)

x=LSTM(128,return_sequences=True)(x)
x=MaxPooling1D()(x)

x=Flatten()(x)

x=Dense(64)(x)
x=LeakyReLU()(x)
x=Dropout(0.4)(x)

x=Dense(128)(x)
x=LeakyReLU()(x)
x=Dropout(0.4)(x)

x=Dense(256, name='dense3')(x)
x=LeakyReLU()(x)
x=Dropout(0.4)(x)

x=Dense(512, name='dense23')(x)
x=LeakyReLU()(x)
x=Dropout(0.4)(x)

x=Dense(1024, name='dense33')(x)
x=LeakyReLU()(x)
x=Dropout(0.4)(x)

x=Dense(2048, name='dense43')(x)
x=LeakyReLU()(x)

output=Dense(6, activation='softmax')(x)

model = Model(input_img, output)

adam=Adam(lr=0.00001)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

model.summary()

from keras.callbacks import TensorBoard

es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto',baseline=None, restore_best_weights=True)

model.fit(X_train, y_train,
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=[es])
model.save('LSTM_model.h5')
