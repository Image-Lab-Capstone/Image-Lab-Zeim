import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_dat import get_scans, pre_process_scans

# LOCATION OF THE FOLDERS CONTAINING THE DICOM FILES.
DATA_FOLDER = 'data/dicom/'
NUM_CLASSES = 10
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

scans = get_scans(DATA_FOLDER)
scans = pre_process_scans(scans)
labels = np.random.randint(0, NUM_CLASSES, len(scans))

# For now just work with a single slice in the scan
scans = scans[:, 0]

x_train, x_test, y_train, y_test = train_test_split(scans, labels, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)
