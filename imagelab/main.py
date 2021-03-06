import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from imagelab.load_data import get_scans
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import argparse
import os
import os.path as osp
import pandas as pd
import pickle
from imagelab.constants import *

def get_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wb', action='store_true')
    parser.add_argument('--viz-model', action='store_true')
    parser.add_argument('--wb-proj', type=str, default='andrew-random')
    parser.add_argument('--image-folders', type=str, default=None)
    parser.add_argument('--image-labels', type=str, default=None)
    parser.add_argument('--load-model-name', type=str, default=None)
    parser.add_argument('--save-name', type=str, default='def')
    parser.add_argument('--manual-crop', type=bool, default=False)
    args = parser.parse_args()
    assert args.image_labels is not None
    assert args.image_folders is not None

    return args

def init_seeds(seed):
    """
    Initializes the random number generator across tensorflow and numpy for
    consistent results across runs.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_data(args):
    """
    Load the image scans into tensorflow dataset. Randomize the datasets and
    load from the location specified in the command line arguments.
    """
    data_folders = args.image_folders.split(',')

    X, y, info = get_scans(data_folders, args.image_labels,
            args.load_model_name is not None,
            args, debug_mode=False)

    X = np.array(X)
    img_width, img_height = X.shape[1:]
    X = X.reshape(-1, img_width, img_height, 1)
    y = np.array(y)
    return X, y, info


def train_model(model, X, y, args, info):
    with open(osp.join(MODELS_DIR, f"{args.save_name}_labels.pickle"), "wb") as f:
        pickle.dump(info['labels'], f)

    SHUFFLE_BUFFER_SIZE = 100

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)

    save_model_file_name = osp.join(MODELS_DIR, args.save_name + '_weights.{epoch:02d}.hdf5')

    callbacks = []
    if args.use_wb:
        # Callback to log results to the W&B logging service.
        callbacks.append(WandbCallback())

    # Callback to save the model every several epochs.
    model_save_cb = tf.keras.callbacks.ModelCheckpoint(save_model_file_name)
    callbacks.append(model_save_cb)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    # Start the training.
    model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)
    model.evaluate(test_dataset)

def evaluate_model(model, X, y, args, info):
    model = load_model(args.load_model_name)
    preds = model.predict_classes(X)
    label_strs = [info['labels'][pred] for pred in preds]
    f_names = [x[1] for x in info['file_names']]
    all_dat = list(zip(f_names, label_strs))
    df = pd.DataFrame(all_dat, columns=['file_name', 'prediction'])
    save_name = args.load_model_name.split('/')[1].replace('.', '_')
    save_path = osp.join(PRED_DIR, save_name + '.csv')
    print(f"Saved results to {save_path}")
    df.to_csv(save_path, index=False)

def construct_model(info, args):
    """
    Construct the Keras deep neural network and train it on the input data.
    Training parameters are specified via `args`.
    """
    if args.use_wb:
        # Logging service used to track training and experiments.
        # For more information about W&B go to https://wandb.ai
        from wandb.keras import WandbCallback
        import wandb
        wandb.init(project=args.wb_proj)
    num_labels = info['uniq_count']
    img_height, img_width = info['data_shape']

    # Define the deep neural netwrok.
    model = tf.keras.Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_width, img_height, 1)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_labels)
    ])

    # Set up the directories we will be saving to (they need to exist).
    if not osp.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not osp.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not osp.exists(PRED_DIR):
        os.makedirs(PRED_DIR)

    # Save a diagram of the model.
    if args.viz_model:
        plot_model(model, to_file=osp.join(DATA_DIR, 'model.png'), show_shapes=True)

    return model

def run():
    """
    Entry point for the entire program. Runs through the process of loading in
    the data and training the model.
    """
    args = get_args()
    init_seeds(args.seed)
    X, y, info = load_data(args)
    model = construct_model(info, args)

    if args.load_model_name is not None:
        evaluate_model(model, X, y, args, info)
    else:
        train_model(model, X, y, args, info)
