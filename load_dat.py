import pydicom
import os
import os.path as osp

DATA_FOLDER = 'data/dicom/38610/'

dcms = []

for f in os.listdir(DATA_FOLDER):
    full_f = osp.join(DATA_FOLDER, f)
    dcm = pydicom.read_file(full_f)
    dcms.append(dcm)


# Do something

