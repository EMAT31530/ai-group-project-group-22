#!/usr/bin/env python
# coding: utf-8

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())


model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', input_shape=(train_data.shape[1:])))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())



model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001, metrics=['accuracy'])

