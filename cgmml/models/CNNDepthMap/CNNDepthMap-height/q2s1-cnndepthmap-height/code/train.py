import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import os
import logging
import glob2 as glob
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks, optimizers
import numpy as np
import pickle
import random
from preprocessing import preprocess_depthmap, preprocess_targets
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

# Parse command line arguments.
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument('--split_seed', nargs=1, default=0, type=int, help="The random seed for splitting.")
parser.add_argument('--target_size', nargs=1, default="172x224", type=str,
                    help="The target image size format WIDTHxHEIGHT.")
parser.add_argument('--epochs', nargs=1, default=1000, type=int, help="The number of epochs.")
parser.add_argument('--batch_size', nargs=1, default=256, type=int, help="The batch size for training.")

args = parser.parse_args()

# Get the split seed.
split_seed = args.split_seed[0]

# Get the image target size.
target_size = args.target_size[0].split("x")
image_target_width = int(target_size[0])
image_target_height = int(target_size[1])

# Get batch size and epochs
batch_size = args.batch_size
epochs = args.epochs

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logger.info('Running in offline mode...')

    # Access workspace.
    logger.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")

    # Get dataset.
    logger.info('Accessing dataset...')
    if os.path.exists("dataset") == False:
        dataset_name = "anon-depthmap-npy"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='dataset', overwrite=False)
    dataset_path = "dataset"

# Online run. Use dataset provided by training notebook.
else:
    logger.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the QR-code paths.
logger.info('Dataset path: %s', dataset_path)
logger.info(glob.glob(os.path.join(dataset_path, "*")))  # Debug
logger.info('Getting QR-code paths...')
qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))

# Shuffle and split into train and validate.
random.seed(split_seed)
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]
del qrcode_paths

# Show split.
logger.info('Paths for training: \n\t' + '\n\t'.join(qrcode_paths_training))
logger.info('Paths for validation: \n\t' + '\n\t'.join(qrcode_paths_validate))


assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
logger.info('Getting depthmap paths...')
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)
del qrcode_paths_training
del qrcode_paths_validate
logger.info('Using %d files for training.', len(paths_training))
logger.info('Using %d files for validation.', len(paths_validate))

# Function for loading and processing depthmaps.


def tf_load_pickle(path):

    def py_load_pickle(path):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = tf.image.resize(depthmap, (image_target_height, image_target_width))
        targets = preprocess_targets(targets, targets_indices)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path], [tf.float32, tf.float32])
    depthmap.set_shape((image_target_height, image_target_width, 1))
    targets.set_shape((len(targets_indices,)))
    return depthmap, targets


def tf_flip(image):

    image = tf.image.random_flip_left_right(image)
    return image


# Parameters for dataset generation.
shuffle_buffer_size = 64
subsample_size = 1024
channels = list(range(0, 3))
targets_indices = [0]  # 0 is height, 1 is weight.

# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path))
dataset = dataset.cache()
dataset = dataset.shuffle(shuffle_buffer_size)
dataset = dataset.map(lambda image, label: (tf_flip(image), label))
#dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_training = dataset
del dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validate = dataset
del dataset


# Note: Now the datasets are prepared.

# Instantiate model.
model = models.Sequential()

model.add(
    layers.Conv2D(
        filters=16,
        kernel_size=(
            3,
            3),
        padding="same",
        activation="relu",
        input_shape=(
                image_target_height,
                image_target_width,
            1)))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation="linear"))
model.summary()


# Get ready to add callbacks.
training_callbacks = []

# Pushes metrics and losses into the run on AzureML.


class AzureLogCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                run.log(key, value)


training_callbacks.append(AzureLogCallback())

# Add TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq="epoch"
)
training_callbacks.append(tensorboard_callback)

# Add checkpoint callback.
best_model_path = "best_model.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

# Compile the model.
model.compile(
    optimizer="nadam",
    loss="mse",
    metrics=["mae"]
)


model.fit(
    dataset_training.batch(batch_size),
    validation_data=dataset_validate.batch(batch_size),
    epochs=epochs,
    callbacks=training_callbacks
)

run.upload_file(name=best_model_path, path_or_stream=best_model_path)

# Save the weights.
#logger.info('Saving and uploading weights...')
#path = "cnndepthmap_weights.h5"
#model.save_weights(path)
#run.upload_file(name="cnndepthmap_weights.h5", path_or_stream=path)

# Save the model.
#logger.info('Saving and uploading model...')
#path = "cnndepthmap_model"
#model.save(path)
#run.upload_folder(name="cnndepthmap", path=path)


# Done.
run.complete()
