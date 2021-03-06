import os
import os.path as osp
import numpy as np
from skimage import io
import cv2
import pandas as pd
from tqdm import tqdm
import imagelab.crop_util
from imagelab.constants import *
import pickle

RESIZE_DIM = 128

def get_scans(img_locs, labels_path, is_eval, args, crop_mag = 50,
        debug_mode=False):
    '''
    manual_crop: Flag to set whether or not doctor selects region of interest
    crop_mag: how severely to decreaser intesity of regions outside of interest
    Returns the following all as a tuple:
        - Image data in numpy format
        - Integer labels in numpy format
        - dictionary: Key 'uniq_count' is the number of image labels.
    '''
    df = pd.read_csv(labels_path)
    def extract_label(x):
        # Only get the first label for now.
        return x.split('|')[0]

    if not is_eval:
        print('TRAINING MODE')
        label_arr = df['Finding Labels'].tolist()
    else:
        print('PREDICTION MODE')
        load_path = osp.join(MODELS_DIR, f"{args.save_name}_labels.pickle")
        if not osp.exists(load_path):
            raise ValueError(f"Expected labels file at {load_path}")

        print(f"Loading labels from {load_path}")
        with open(load_path, "rb") as f:
            uniq_vals = pickle.load(f)
        # Use just garbage labels
        label_arr = [uniq_vals[0] for _ in range(len(df))]

    label_map = {k: extract_label(v) for k, v in zip(df['Image Index'].tolist(), label_arr)}

    if not is_eval:
        # Get the set of unique labels.
        uniq_vals = list(set(list(label_map.values())))

    X = []
    y = []

    all_file_paths = []
    # Get the paths we need to load images from.
    for img_loc in img_locs:
        print('Loading files from %s' % img_loc)
        take_file_paths = os.listdir(img_loc)
        all_file_paths.extend([(img_loc, fp) for fp in take_file_paths])

    if debug_mode:
        # Optionally only load part of the files.
        all_file_paths = all_file_paths[:20]

    # Load in all the images.
    for img_loc, scan_path in tqdm(all_file_paths):
        full_path = osp.join(img_loc, scan_path)
        im = io.imread(full_path)
        if len(im.shape) == 3:
            # Some images are weird and have a color channel associated with
            # them.
            continue
        # Resize each image to a fixed shape.
        im = cv2.resize(im, (RESIZE_DIM, RESIZE_DIM))

        #check if user wants to crop for region of interest
        if(args.manual_crop):
            im  = imagelab.crop_util.select_roi(im, crop_mag)
        label_str = label_map[scan_path]
        label = uniq_vals.index(label_str)

        X.append(im)
        y.append(label)

    print('Loaded %i images' % len(X))

    return X, y, {
            'label_map': label_map,
            'labels': uniq_vals,
            'file_names': all_file_paths,
            'uniq_count': len(uniq_vals),
            'data_shape': X[0].shape,
            }


if __name__ == '__main__':
    get_scans('/data/aszot/kaggle/images_001/images', '/data/aszot/kaggle/Data_Entry_2017.csv')
