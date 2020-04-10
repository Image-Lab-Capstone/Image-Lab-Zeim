#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:30:08 2020

@author: noahforman
"""
import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




def proc_images():
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    
    disease="Infiltration"
    disease2="Atelectasis"
    # ADD the other diseases here #

    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 128
    HEIGHT = 128

    for img in images:
        base = os.path.basename(img)
        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]

        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

        # Labels
        if disease in finding:
            #finding = str(disease)
            finding = 1
            y.append(finding)
        elif disease2 in finding:
            finding = 2
            y.append(finding)
        # Add the other disease conditions here
        else:
            #finding = "Not_" + str(disease)
            finding = 0
            y.append(finding)

    return x,y






PATH = os.path.abspath(os.path.join("..", 'input')) # Add your own filepath here

# ../input/sample/images/
SOURCE_IMAGES = os.path.join(PATH, "sample", "images")

# ../input/sample/images/*.png
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

# Load labels
labels = pd.read_csv('../input/sample_labels.csv') # Add your own filepath here

# First five images paths
print(images[0:5])

r = random.sample(images, 3)
r

# Matplotlib black magic
plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.imread(r[0]))

plt.subplot(132)
plt.imshow(cv2.imread(r[1]))

plt.subplot(133)
plt.imshow(cv2.imread(r[2]));  

x,y = proc_images()

# Set it up as a dataframe if you like
df = pd.DataFrame()
df["labels"]=y
df["images"]=x

print(len(df), df.images[0].shape)

np.savez("x_images_arrays", x)
np.savez("y_infiltration_labels", y)























