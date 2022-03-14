import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tqdm import tqdm

src_path = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//train"

sub_class = os.listdir(src_path)

TRAIN_SIZE = 60000

train_df = pd.read_csv('train_labels.csv')
test_df = pd.read_csv('test_labels.csv')[:16000]

train_neg = train_df[train_df['label']==0].sample(TRAIN_SIZE,random_state=45)
train_pos = train_df[train_df['label']==1].sample(TRAIN_SIZE,random_state=45)

train_df = pd.concat([train_neg, train_pos], axis=0).reset_index(drop=True)

train_df['label'] = train_df['label'].astype(str)



def append_ext(fn):
    return fn+".tif"

train_df['id'] = train_df['id'].apply(append_ext)
test_df['id'] = test_df['id'].apply(append_ext)

print(train_df.head())


src_path_train = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//train//"
src_path_test = "C://Users//OliWo//OneDrive - University of Bristol//Intro to AI//Data//test//"

train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

batch_size = 64
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=src_path_train,
    x_col="id",
    y_col="label",
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=src_path_train,
    x_col="id",
    y_col="label",
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=src_path_test,
    x_col="id",
    target_size=(100, 100),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

def prepare_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))#, input_shape=(100, 100, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(tf.keras.layers.Dropout(0.2))
    #

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#, input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(tf.keras.layers.Dropout(0.3))
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))  # , input_shape=(100, 100, 3)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))  # , input_shape=(32, 32, 3)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))  # , input_shape=(100, 100, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.BatchNormalization())
    #
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))  # , input_shape=(32, 32, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(Flatten())
    #model.add(Dense(512, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

    # fit model here

    return model

model = prepare_model()
model.fit_generator(train_generator,
                    validation_data = train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = valid_generator.n//valid_generator.batch_size,
                    epochs=10)

score = model.evaluate_generator(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict=model.predict_generator(test_generator, steps = len(test_generator.filenames))

y_classes = predict.argmax(axis=-1)

test_labels = np.array(test_df['label'])

print(test_labels)

df = [y_classes,test_labels]#, np.array(valid_df['id'])]
df = pd.DataFrame(df)#, np.array(valid_df['id'])]
df_t = df.T

pd.DataFrame(df_t).to_csv('output_4layers_dropout_0.3.csv')