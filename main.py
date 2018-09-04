# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:45:29 2018
Nivia model

@author: DELL
"""

import os
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import cv2
import pandas as pd
import random
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_image_steering(path, df):
    image_path = []
    steering = []
    for i in range(len(df)):
        data = df.iloc[i]
        center = data[0]
        steering.append(data[3])
        image_path.append(os.path.join(path, center.strip()))
    image_path = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_path, steering


def preprocessing_image(img_path):
    img = mpimg.imread(img_path)
    img = img[60:140, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def nivia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5,5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu'))
    model.add(Convolution2D(36, (5,5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, (5,5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    model.compile('adam', 'mse', ['accuracy'])
    return model


columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv('data/driving_log.csv', names = columns)
pd.set_option('display.max_colwidth', -1)

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

num_bin = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bin)
center = (bins[:-1] + bins[1:])/2
plt.bar(center, hist, width=0.05)

remove_list = []
for i in range(num_bin):
    lst_redundent = []
    for j in range(len(data['steering'])):
        if data['steering'][j] >= bins[i] and data['steering'][j] <= bins[i+1]:
            lst_redundent.append(j)
    shuffle(lst_redundent)
    lst_redundent = lst_redundent[samples_per_bin:]
    remove_list.extend(lst_redundent)

# drop data to make uniform
data.drop(data.index[remove_list], inplace = True)

hist, bins = np.histogram(data['steering'], num_bin)
center = (bins[:-1] + bins[1:])/2
plt.bar(center, hist, width=0.05)

image_path, steering = load_image_steering('Data/IMG/', data)

# Seperate traing set and test set
X_train, X_valid, y_train, y_valid = train_test_split(image_path, steering, test_size = 0.2)
fig, axes = plt.subplots(1, 2, figsize = (12, 4))
axes[0].hist(y_train, bins=num_bin, width=0.05, color='blue')
axes[0].set_title('training set')
axes[1].hist(y_valid, bins=num_bin, width=0.05, color='red')
axes[1].set_title('validation set')

# Preprocess image

image=image_path[100]
original_image = mpimg.imread(image)
processed_image = preprocessing_image(image)
fig, axes = plt.subplots(1, 2, figsize = (12, 4))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original image')
axes[1].imshow(processed_image)
axes[1].set_title('Processed image')

X_train = np.array(list(map(preprocessing_image, X_train)))
X_valid = np.array(list(map(preprocessing_image, X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')

model = nivia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=50, verbose=1, shuffle=1)
