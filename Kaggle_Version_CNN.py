# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:31:29 2022

@author: matth
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from numpy.random import seed
seed(42)
import pickle

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob 
#matplotlib inline

IMG_SIZE = 96
IMG_CHANNELS = 3
#TRAIN_SIZE=80000
TRAIN_SIZE = 8000
BATCH_SIZE = 64
EPOCHS = 10


#I used cropped images
train_labels = pd.read_csv('train_labels.csv')
train_labels['label'].value_counts()


(train_labels.label.value_counts() / len(train_labels)).to_frame()


train_neg = train_labels[train_labels['label']==0].sample(TRAIN_SIZE,random_state=45)
train_pos = train_labels[train_labels['label']==1].sample(TRAIN_SIZE,random_state=45)

train_data = pd.concat([train_neg, train_pos], axis=0).reset_index(drop=True)

train_data = shuffle(train_data)


train_data['label'].value_counts()


def append_ext(fn):
    return fn+".tif"


y = train_data['label']
train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=45, stratify=y)

print(train_df.shape)
print(valid_df.shape)


train_df['id'] = train_df['id'].apply(append_ext)
valid_df['id'] = valid_df['id'].apply(append_ext)
train_df.head()


train_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)


BATCH_SIZE = 64
train_path = './train'
train_df['label'] = train_df['label'].astype(str)
valid_df['label'] = valid_df['label'].astype(str)

train_loader = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_path,
    x_col = 'id',
    y_col = 'label',
    batch_size = BATCH_SIZE,
    seed = 42,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (32,32)
)

valid_loader = valid_datagen.flow_from_dataframe(
    dataframe = valid_df,
    directory = train_path,
    x_col = 'id',
    y_col = 'label',
    batch_size = BATCH_SIZE,
    seed = 42,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (32,32)
)


TR_STEPS = len(train_loader)
VA_STEPS = len(valid_loader)

print(TR_STEPS)
print(VA_STEPS)


cnn = Sequential([
    Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape=(32,32,3)),
    Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.4),
    BatchNormalization(),

    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    
    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.6),
    BatchNormalization(),

    Flatten(),
     Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.4),
    Dense(8, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(2, activation='softmax')
])


opt = tf.keras.optimizers.Adam(0.001)
cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])



h1 = cnn.fit(
    x = train_loader, 
    steps_per_epoch = TR_STEPS, 
    epochs = 30,
    validation_data = valid_loader, 
    validation_steps = VA_STEPS, 
    verbose = 1
)



tf.keras.backend.set_value(cnn.optimizer.learning_rate, 0.0001)



h2 = cnn.fit(
    x = train_loader, 
    steps_per_epoch = TR_STEPS, 
    epochs = 30,
    validation_data = valid_loader, 
    validation_steps = VA_STEPS, 
    verbose = 1
)



tf.keras.backend.set_value(cnn.optimizer.learning_rate, 0.00001)



h3 = cnn.fit(
    x = train_loader, 
    steps_per_epoch = TR_STEPS, 
    epochs = 30,
    validation_data = valid_loader, 
    validation_steps = VA_STEPS, 
    verbose = 1
)