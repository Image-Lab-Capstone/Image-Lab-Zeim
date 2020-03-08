#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:01:05 2020

@author: noahforman
"""
from __future__ import absolute_import, division, print_function, unicode_literals
 
import numpy as np
import tensorflow as tf


train_dataset = tf.data.Dataset.from_tensor_slices(( , )) #(train_examples, train_labels)
test_dataset = tf.data.Dataset.from_tensor_slices(( , )) #(test_examples, test_labels)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense() # This will need to change depending on the number of classes we have.
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)
