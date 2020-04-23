import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from imagelab.load_data import get_scans
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import argparse
import os
import os.path as osp

# Predefined directories for models and data. Models and images must be placed
# within these directories. Please see the documentation for more information.
MODELS_DIR = 'trained_models/'
DATA_DIR = 'data/'


def get_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wb', action='store_true')
    parser.add_argument('--wb-proj', type=str, default='andrew-random')
    parser.add_argument('--image-folders', type=str, default=None)
    parser.add_argument('--image-labels', type=str, default=None)
    args = parser.parse_args()
    assert args.image_labels is not None
    assert args.image_folders is not None

    return args

def init_seeds():
    """
    Initializes the random number generator across tensorflow and numpy for
    consistent results across runs.
    """
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

def load_data(args):
    """
    Load the image scans into tensorflow dataset. Randomize the datasets and
    load from the location specified in the command line arguments.
    """
    data_folders = args.image_folders.split(',')

    SHUFFLE_BUFFER_SIZE = 100

    X, y, info = get_scans(data_folders, args.image_labels, debug_mode=False)

    X = np.array(X)
    img_width, img_height = X.shape[1:]
    X = X.reshape(-1, img_width, img_height, 1)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)
    return train_dataset, test_dataset, info

def train_model(train_dataset, test_dataset, info, args):
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

    # Save a diagram of the model.
    plot_model(model, to_file=osp.join(DATA_DIR, 'model.png'), show_shapes=True)

    save_model_file_name = osp.join(MODELS_DIR, 'weights.{epoch:02d}.hdf5')

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

def main():
    """
    Entry point for the entire program. Runs through the process of loading in
    the data and training the model.
    """
    init_seeds()
    args = get_args()
    train_dataset, test_dataset, info = load_data(args)
    train_model(train_dataset, test_dataset, info, args)
