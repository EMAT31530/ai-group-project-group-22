# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:23:09 2022

@author: matth
"""
import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import patches
import cv2

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
# from tensorflow.keras import backend as K
# K.set_image_dim_ordering('th')


path_train = 'C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\train'
path_test = 'C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\test'

train = listdir(path_train)
test = listdir(path_test)

#Checking the paths work
# A = train[:5]
# B = test[:5]
# print(A)
# print()
# print(B)
# print(type(train))

def load_train():
    train_data = []
    for img in os.listdir(path_train):
        img_array = cv2.imread('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\train\\' +img, cv2.IMREAD_COLOR)
        train_data.append(img_array)
    train_data = np.array(train_data)
        
    return train_data
        
train_data = load_train()

def load_test():
    test_data = []
    for img in os.listdir(path_test):
        img_array = cv2.imread('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\test\\' +img, cv2.IMREAD_COLOR)
        test_data.append(img_array)
    test_data = np.array(test_data)
        
    return test_data
        
test_data = load_test()
test_data = test_data[0:2000]

print(train_data.shape)
print(test_data.shape)

train_labels = pd.read_csv('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\data_labels_train.csv')['label'].tolist()
train_labels = np.array(train_labels)
#Given data download messed up have to change the labels size
train_labels = train_labels[:9000]

test_labels = pd.read_csv('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\data_labels_test.csv')['label'].tolist()
test_labels = np.array(test_labels)
#Given data download messed up have to change the labels size
test_labels = test_labels[:2000]


#Checking the paths work
print('The first 5 images from train_data are: ', train[:5])
print('And their labels are: ', train_labels[:5])
print()
print('The first 5 images from test_data are: ', test[:5])
print('And their labels are: ', test_labels[:5])
# #print(type(train))

#So we now have train_data, train_labels, test_data and test_labels

#Now create alexnet CNN

#def alexnet(train_data, test_data, train_labels, test_labels):
  # model = tf.keras.models.Sequential([
  #   tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape = (train_data.shape[1:])),
  #   tf.keras.layers.Lambda(tf.nn.local_response_normalization),
  #   #NEED TO CHANGE POOL SIZE FROM 3 TO 2
  #   tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    # tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    # tf.keras.layers.Lambda(tf.nn.local_response_normalization),
    # tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    # tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # tf.keras.layers.Lambda(tf.nn.local_response_normalization),
    # tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # tf.keras.layers.Lambda(tf.nn.local_response_normalization),
    # tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # tf.keras.layers.Lambda(tf.nn.local_response_normalization),
    # tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
  #   tf.keras.layers.Flatten(),
  #   tf.keras.layers.Dense(4096, activation='relu'),
  #   tf.keras.layers.Dropout(0.5),
  #   tf.keras.layers.Dense(4096, activation='relu'),
  #   tf.keras.layers.Dropout(0.5),
  #   tf.keras.layers.Dense(2, activation='softmax')
  # ])
  
  #HAD TO ADD PADDING TO THE MAXPOOLING LAYERS https://github.com/keras-team/keras/issues/3945

def alexnet(train_data, test_data, train_labels, test_labels):
  model = Sequential()
  model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape = (train_data.shape[1:])))
  model.add(Lambda(tf.nn.local_response_normalization))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))# data_format="channels_first"))
  model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation='relu', padding="same"))
  model.add(Lambda(tf.nn.local_response_normalization))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))# data_format="channels_first"))
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  model.add(tf.keras.layers.Lambda(tf.nn.local_response_normalization))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))# data_format="channels_first"))
  
  model.add(Flatten())
  model.add(Dense(4096, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation="relu"))
  model.add(Dropout(0.5))
  # model.add(Dense(2048, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(1024, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(512, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(266, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(133, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(50, activation="relu"))
  # model.add(Dropout(0.5))
  # model.add(Dense(25, activation="relu"))
  # model.add(Dropout(0.5))
  #model.add(Dense(2, activation="relu"))
  model.add(Dense(1, activation='sigmoid'))
  
  # compile model here
  model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

  # fit model here
  model.fit(train_data, train_labels, epochs=10)

  # evaluate model on test set here
  results = model.evaluate(test_data, test_labels)
  print(results)
  return model

model = alexnet(train_data, test_data, train_labels, test_labels)




