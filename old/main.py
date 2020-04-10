#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Code from: https://www.tensorflow.org/tutorials/images/classification
# ---------------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import preprocessing

#_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'  # EDITED THIS LINE

#path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)  # EDITED THIS LINE

#PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')  # EDITED THIS LINE

train_dir = os.path.join(, 'train') # FIRST PARAMETER IS FILEPATH TO INPUT DATA -- WILL BE DIFFERENT ON EACH COMPUTER
validation_dir = os.path.join(, 'validation') # FIRST PARAMETER IS FILEPATH TO INPUT DATA -- WILL BE DIFFERENT ON EACH COMPUTER

train_head_dir = os.path.join(train_dir, 'head')  # directory with our training cat pictures
train_hip_dir = os.path.join(train_dir, 'hip')  # directory with our training dog pictures
train_pelvis_dir = os.path.join(train_dir, 'pelvis') # EDITED THIS LINE
train_shoulder_dir = os.path.join(train_dir, 'shoulder') # EDITED THIS LINE
validation_head_dir = os.path.join(validation_dir, 'head')  # directory with our validation cat pictures
validation_hip_dir = os.path.join(validation_dir, 'hip')  # directory with our validation dog pictures
validation_pelvis_dir = os.path.join(validation_dir, 'pelvis') # EDITED THIS LINE
validation_shoulder_dir = os.path.join(validation_dir, 'shoulder') # EDITED THIS LINE

num_head_tr = len(os.listdir(train_head_dir))
num_hip_tr = len(os.listdir(train_hip_dir))
num_pelvis_tr = len(os.listdir(train_pelvis_dir)) # EDITED THIS LINE
num_shoulder_tr = len(os.listdir(train_shoulder_dir)) # EDITED THIS LINE

num_head_val = len(os.listdir(validation_head_dir))
num_hip_val = len(os.listdir(validation_hip_dir))
num_pelvis_val = len(os.listdir(validation_pelvis_dir)) # EDITED THIS LINE
num_shoulder_val = len(os.listdir(validation_shoulder_dir)) # EDITED THIS LINE

total_train = num_head_tr + num_hip_tr + num_pelvis_tr + num_shoulder_tr # EDITED THIS LINE
total_val = num_head_val + num_hip_val + num_pelvis_val + num_shoulder_val # EDITED THIS LINE

print('total training head images:', num_head_tr)
print('total training hip images:', num_hip_tr)

print('total validation head images:', num_head_val)
print('total validation hip images:', num_hip_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 30
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           horizontal_flip=True,
                                           rotation_range=45,
                                           zoom_range=0.5,
                                           width_shift_range=0.15,
                                           height_shift_range=0.15) # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')




#sample_training_images, _ = next(train_data_gen)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(sample_training_images[:5])
plotImages(augmented_images)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax') # EDITED THIS LINE
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # EDITED THIS LINE
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()















