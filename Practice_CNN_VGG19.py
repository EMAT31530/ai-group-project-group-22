# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:39:33 2022

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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential


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

print(train_data.shape)
print(test_data.shape)

train_labels = pd.read_csv('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\data_labels_train.csv')['label'].tolist()
train_labels = np.array(train_labels)
#Given data download messed up have to change the labels size
train_labels = train_labels[:9000]

test_labels = pd.read_csv('C:\\Users\\matth\\OneDrive - University of Bristol\\Documents Year 4\\Introduction to Artificial Intelligence\\Group Project\\Data\\data_labels_test.csv')['label'].tolist()
test_labels = np.array(test_labels)
#Given data download messed up have to change the labels size
test_labels = test_labels[:9000]


#print(test_labels[:5])

#So we no have train_data, train_labels, test_data and test_labels


def conv_block(channels, kernel_size=(3,3), activation='relu',use_bn=True):
  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, activation=None, padding='same'))
  if use_bn:
    model.add(tf.keras.layers.BatchNormalization())
  
  if activation=='relu':
    model.add(tf.keras.layers.ReLU())
  return model

def vgg19(train_data, test_data, train_labels, test_labels):
  #Build model here
  model = Sequential()
  model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape = (train_data.shape[1:])))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D())  
  #input_image = tf.keras.layers.Input(train_data.shape[1:])
  #model = conv_block(64, (3,3), 'relu', use_bn=True)(input_image)
  #model = conv_block(64, (3,3), 'relu', use_bn=True)(model)
  #model = tf.keras.layers.MaxPool2D()(model)

  model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D())  
  # model = conv_block(128, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(128, (3,3), 'relu', use_bn=True)(model)
  # model = tf.keras.layers.MaxPool2D()(model)

  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D()) 
  # model = conv_block(256, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(256, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(256, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(256, (3,3), 'relu', use_bn=True)(model)
  # model = tf.keras.layers.MaxPool2D()(model)

  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D()) 
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = tf.keras.layers.MaxPool2D()(model)

  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same',))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling2D()) 
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = conv_block(512, (3,3), 'relu', use_bn=True)(model)
  # model = tf.keras.layers.MaxPool2D()(model)

  model.add(Flatten())
  model.add(Dense(4096))
  model.add(Dense(4096))
  model.add(Dense(1000, activation='softmax'))
  # model = tf.keras.layers.Flatten()(model)
  # model = tf.keras.layers.Dense(4096, activation='relu')(model)
  # model = tf.keras.layers.Dense(4096, activation='relu')(model)
  # model = tf.keras.layers.Dense(1000, activation='softmax')(model)

  
  #required to merge models
  
  
  # compile model here
  model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

  # fit model here
  model.fit(train_data, train_labels, epochs=3)

  # evaluate model on test set here
  results = model.evaluate(test_data, test_labels)
  print(results)
  return model #tf.keras.Model(input_image, model)

model = vgg19(train_data, test_data, train_labels, test_labels)