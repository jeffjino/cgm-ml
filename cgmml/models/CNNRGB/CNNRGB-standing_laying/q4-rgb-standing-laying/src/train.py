from pathlib import Path
import os
import random
import logging
import numpy as np

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import wandb
from wandb.keras import WandbCallback

from cgmml.common.model_utils.utils import (
    download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback, setup_wandb)
from sl_config import CONFIG
from sl_constants import MODEL_CKPT_FILENAME, REPO_DIR
from model import create_cnn, set_trainable_below_layers
from sl_preprocessing import process_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)
logger.info('infotest')

run = Run.get_context()

# Make experiment reproducable
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")
logger.info('DATA_DIR: %s', DATA_DIR)

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logger.info('Running in offline mode...')

    # Access workspace.
    logger.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    # Get dataset.
    logger.info('Accessing dataset...')
    dataset_name = "anon-rgb-classification"
    dataset_path = str(REPO_DIR / "data" / dataset_name)
    if not os.path.exists(dataset_path):
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path=dataset_path, overwrite=False)

    dataset_name = CONFIG.DATASET_NAME_LOCAL
    dataset_path = get_dataset_path(DATA_DIR, dataset_name)
    download_dataset(workspace, dataset_name, dataset_path)

# Online run. Use dataset provided by training notebook.
else:
    logger.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace

    dataset_name = CONFIG.DATASET_NAME
    dataset_path = run.input_datasets['cgm_dataset']


# Get the Image paths.
dataset_path = os.path.join(dataset_path, "train")
logger.info('Dataset path: %s', dataset_path)
logger.info('Getting image...')
image_paths = glob.glob(os.path.join(dataset_path, "*/*.jpg"))
logger.info('Image Path: % d', len(image_paths))
assert len(image_paths) != 0

# Shuffle and split into train and validate.
random.shuffle(image_paths)
split_index = int(len(image_paths) * 0.8)
image_paths_training = image_paths[:split_index]
image_paths_validate = image_paths[split_index:]

del image_paths

# Show split.

logger.info('Paths for training: \n\t' + '\n\t'.join(image_paths_training))
logger.info('Paths for validation: \n\t' + '\n\t'.join(image_paths_validate))

logger.info('Nbr of image_paths for training: %d', len(image_paths_training))
logger.info('Nbr of image_paths for validation: %d', len(image_paths_validate))

assert len(image_paths_training) > 0 and len(image_paths_validate) > 0

# Parameters for dataset generation.
class_names = np.array(sorted([item.split('/')[-1] for item in glob.glob(os.path.join(dataset_path, "*"))]))
print(class_names)

# Create dataset for training.
paths = image_paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: process_path(path))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset
del dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = image_paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: process_path(path))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset
del dataset


def create_and_fit_model():
    input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3)
    model = create_cnn(input_shape, dropout=True)
    model.summary()
    logger.info(len(model.trainable_weights))

    # Add checkpoint callback.
    #best_model_path = os.path.join('validation','best_model.h5')
    best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    dataset_batches = dataset_training.batch(CONFIG.BATCH_SIZE)

    training_callbacks = [
        AzureLogCallback(run),
        create_tensorboard_callback(),
        checkpoint_callback,
    ]
    if getattr(CONFIG, 'USE_WANDB', False):
        setup_wandb()
        wandb.init(project="ml-project", entity="cgm-team")
        wandb.config.update(CONFIG)
        training_callbacks.append(WandbCallback(log_weights=True, log_gradients=True, training_data=dataset_batches))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=CONFIG.LEARNING_RATE)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # Train the model.
    model.fit(
        dataset_training.batch(CONFIG.BATCH_SIZE),
        validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
        epochs=CONFIG.EPOCHS,
        callbacks=training_callbacks,
        verbose=2
    )

    #  function use to tune the top convolution layer
    set_trainable_below_layers('block14_sepconv1', model)

    model.fit(
        dataset_training.batch(CONFIG.BATCH_SIZE),
        validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
        epochs=CONFIG.TUNE_EPOCHS,
        callbacks=training_callbacks,
        verbose=2
    )


if CONFIG.USE_MULTIGPU:
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: %s", strategy.num_replicas_in_sync)
    with strategy.scope():
        create_and_fit_model()
else:
    create_and_fit_model()

# Done.
run.complete()
