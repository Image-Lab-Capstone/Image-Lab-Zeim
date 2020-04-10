import pydicom
import os
import os.path as osp
import numpy as np
import cv2


def get_scans(data_loc):
    dcms = []
    for dcm_folder in os.listdir(data_loc):
        patient_dcm = []
        for f in os.listdir(osp.join(data_loc, dcm_folder)):
            full_f = osp.join(data_loc, dcm_folder, f)
            dcm_slice = pydicom.read_file(full_f)
            patient_dcm.append(dcm_slice)
        dcms.append(patient_dcm)

    dcm_np = [[dcm.pixel_array for dcm in patient_dcm] for patient_dcm in dcms]

    min_dims = np.min([[x.shape
        for x in patient_dcm]
        for patient_dcm in dcm_np], axis=-1)[0]

    # This will need to change because some patients only have one scan?
    min_scans = min([len(x) for x in dcm_np])

    dcm_np = np.array([[x[:min_dims[0],:min_dims[1]]
        for x in patient_dcm[:min_scans]]
        for patient_dcm in dcm_np])
    return dcm_np


def pre_process_scans(dcm_np):
    # (num patients, number of scans per patient, width, length)
    n, d, w, l = dcm_np.shape

    # The crop dimensions around the center of the image.
    CROP_DIM = (300, 200)

    # Crop
    hl = l // 2
    hw = w // 2
    cropped_dcms = dcm_np[:, :, hw-(CROP_DIM[0]//2):hw+(CROP_DIM[0]//2), hl-(CROP_DIM[1]//2):hl+(CROP_DIM[1]//2)]

    # Normalize
    normalized_dc = (cropped_dcms - np.mean(cropped_dcms, axis=0)) / np.std(cropped_dcms, axis=0)
    resized_dc = np.array([[cv2.resize(x, (100, 100))
            for x in patient_dcm]
            for patient_dcm in normalized_dc])
    return resized_dc



