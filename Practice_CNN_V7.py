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
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, Adamax, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda

from tqdm import tqdm

path_test = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//test"
path_train = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//train"

train = listdir(path_train)
test = listdir(path_test)


def load_train():
    train_data = []
    for img in tqdm(os.listdir(path_train)):
        img_array = cv2.imread("C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//train//" + img, cv2.IMREAD_COLOR)
        train_data.append(img_array)
    train_data = np.array(train_data)

    return train_data


train_data = load_train()
#train_data = train_data[0:9000]


def load_test():
    test_data = []
    for img in os.listdir(path_test):
        img_array = cv2.imread("C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//test//" + img, cv2.IMREAD_COLOR)
        test_data.append(img_array)
    test_data = np.array(test_data)

    return test_data


test_data = load_test()
#test_data = test_data[0:3000]

print(train_data.shape)
print(test_data.shape)

train_labels = pd.read_csv("C:\\Users\\OliWo\\OneDrive - University of Bristol\\Intro to AI\\Data\\train_labels.csv")['label'].tolist()
train_labels = np.array(train_labels)
# Given data download messed up have to change the labels size
#train_labels = train_labels[0:9000]

test_labels = pd.read_csv("C:\\Users\\OliWo\\OneDrive - University of Bristol\\Intro to AI\\Data\\test_labels.csv")['label'].tolist()
test_labels = np.array(test_labels)
# Given data download messed up have to change the labels size
#test_labels = test_labels[0:3000]

# Checking the paths work
print('The first 5 images from train_data are: ', train[:5])
print('And their labels are: ', train_labels[:5])
print()
print('The first 5 images from test_data are: ', test[:5])
print('And their labels are: ', test_labels[:5])


# #print(type(train))

# So we no have train_data, train_labels, test_data and test_labels

# Now create CNN
def build_fit_eval_model(train_data, test_data, train_labels, test_labels):
    height = train_data.shape[1]
    width = train_data.shape[2]
    channels = train_data.shape[3]
    num_classes = 1

    # build model here.
    model = Sequential()

    # model.add(
    #     Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())


    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu'))#, input_shape=(train_data.shape[1:])))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     activation='relu'))#, input_shape=(train_data.shape[1:])))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                     activation='relu'))  # , input_shape=(train_data.shape[1:])))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(tf.keras.layers.BatchNormalization()) #Doesn't help
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # fit model here
    model.fit(train_data, train_labels, epochs=10)

    # evaluate model on test set here
    results = model.evaluate(test_data, test_labels)
    print(results)
    return model


model = build_fit_eval_model(train_data, test_data, train_labels, test_labels)

predictions = model.predict(test_data)

df = pd.DataFrame(predictions)

with open('file2.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(predictions)