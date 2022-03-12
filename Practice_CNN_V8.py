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


TRAIN_SIZE = 500
#TRAIN_SIZE = 1000
BATCH_SIZE = 64

train_labels = pd.read_csv('train_labels.csv')

train_neg = train_labels[train_labels['label']==0].sample(TRAIN_SIZE,random_state=45)
train_pos = train_labels[train_labels['label']==1].sample(TRAIN_SIZE,random_state=45)

train_data = pd.concat([train_neg, train_pos], axis=0).reset_index(drop=True)

train_data = shuffle(train_data)

print(train_data['label'].value_counts())

def append_ext(fn):
    return fn+".tif"

y = train_data['label']
train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=45, stratify=y)


train_df['id'] = train_df['id'].apply(append_ext)
valid_df['id'] = valid_df['id'].apply(append_ext)

train_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)

BATCH_SIZE = 64
#train_path = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//train"
train_path = "C://Users//matth//OneDrive - University of Bristol//Documents Year 4//Introduction to Artificial Intelligence//Group Project//Data//train"
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

print(train_loader)

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


model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    BatchNormalization(),

    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    BatchNormalization(),

    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    BatchNormalization(),

    Conv2D(256, (3,3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    BatchNormalization(),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    #BatchNormalization(),
    Dense(2, activation='softmax') ])


opt = tf.keras.optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])

h1 = model.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = 5,
    validation_data = valid_loader,
    validation_steps = VA_STEPS,
    verbose = 1)

#predicted_test_vals = (model.predict(valid_df))
predict=model.predict(valid_loader, steps = len(valid_loader.filenames))
#print(predict_test_vals)
print(predict)
print(len(predict))

y_classes = predict.argmax(axis=-1)
print(y_classes)

