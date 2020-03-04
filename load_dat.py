import pydicom
import os
import os.path as osp
import numpy as np

# LOCATION OF THE FOLDERS CONTAINING THE DICOM FILES.
DATA_FOLDER = 'data/dicom/'


dcms = []
for dcm_folder in os.listdir(DATA_FOLDER):
    patient_dcm = []
    for f in os.listdir(osp.join(DATA_FOLDER, dcm_folder)):
        full_f = osp.join(DATA_FOLDER, dcm_folder, f)
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


# (num patients, number of scans per patient, width, length)
n, d, w, l = dcm_np.shape

# The crop dimensions around the center of the image.
CROP_DIM = (300, 200)

# Crop
import ipdb; ipdb.set_trace()
cropped_dcms = dcm_np[:, :, w-(CROP_DIM[0]//2):w+(CROP_DIM[0]//2), l-(CROP_DIM[1]//2):l+(CROP_DIM[1]//2)]


# Normalize
normalized_dc = (cropped_dcms - np.mean(cropped_dcms, axis=0)) / np.std(cropped_dcms, axis=0)



