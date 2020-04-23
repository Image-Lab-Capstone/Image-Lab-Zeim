#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Exploring pre-processing techniques for DICOM images. Also starting on the
# initial data pipeline to load in the DICOM images.
#
# This script is based off various DICOM processing tutorials from the Kaggle
# 2017 data science bowl challenge found here:
# https://www.kaggle.com/c/data-science-bowl-2017
# ---------------------------------------------------------------------

import dicom
import scipy.ndimage
import os
import numpy as np
import math
import cv2

from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage import measure, feature
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing

import matplotlib.pyplot as plt

# CONSTANTS FROM: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
IMG_PX_SIZE = 150
HM_SLICES = 20
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    """
    Convert the DICOM scans to numpy arrays
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    outside_image = image.min()
    image[image == outside_image] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(scan, image, new_spacing=[1,1,1]):
    """
    Resize image to a constant, smaller size.
    """
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    batch = []
    for i in range(0, image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = image[i + j]
            img= cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(tmp)

    batch = np.array(batch)

    return batch


def segment_region_of_interest(image):
    """
    Looking into how to perform automated segmentation
    """
    binary = image < 604
    cleared = clear_border(binary)

    label_image = label(cleared)

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0

    binary = label_image > 0

    selem = disk(2)
    binary = binary_erosion(binary, selem)

    selem = disk(10)
    binary = binary_closing(binary, selem)

    edges = roberts(binary)
    binary = scipy.ndimage.binary_fill_holes(edges)

    get_high_vals = binary == 0
    image[get_high_vals] = 0

    return image


def full_preprocess(scan):
    scan_pixels = get_pixels_hu(scan)
    resampled_img = resample(scan, scan_pixels, [1,1,1])
    resampled_img = resampled_img - PIXEL_MEAN

    resampled_img = normalize(resampled_img)

    resampled_image = (resampled_image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    resampled_image[resampled_image > 1] = 1
    resampled_image[resampled_image < 0] = 0
    return resampled_img


scan = load_scan('~/dicom_tests')
resampled_image = full_preprocess(scan)

