import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_data import get_scans
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--use-wb', action='store_true')
parser.add_argument('--wb-proj', type=str, default='andrew-random')
args = parser.parse_args()

if args.use_wb:
    from wandb.keras import WandbCallback
    import wandb
    wandb.init(project=args.wb_proj)

# LOCATION OF THE FOLDERS CONTAINING THE DICOM FILES.
DATA_FOLDER = '/data/aszot/kaggle/images_001/images'
LABEL_PATH = '/data/aszot/kaggle/Data_Entry_2017.csv'
SHUFFLE_BUFFER_SIZE = 100

X, y, info = get_scans(DATA_FOLDER, LABEL_PATH, debug_mode=False)
num_labels = info['uniq_count']

X = np.array(X)
img_width, img_height = X.shape[1:]
X = X.reshape(-1, img_width, img_height, 1)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
test_dataset = test_dataset.batch(args.batch_size)

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


callbacks = []
if args.use_wb:
    callbacks.append(WandbCallback())

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)
model.evaluate(test_dataset)
