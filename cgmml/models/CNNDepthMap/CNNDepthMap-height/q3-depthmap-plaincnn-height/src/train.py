from pathlib import Path
import os
import pickle
import random
import logging

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import wandb
from wandb.keras import WandbCallback

from cgmml.common.model_utils.model_plaincnn import create_cnn
from cgmml.common.model_utils.preprocessing import filter_blacklisted_persons, preprocess_depthmap, preprocess_targets
from cgmml.common.model_utils.utils import (
    download_dataset, get_dataset_path, AzureLogCallback,
    create_tensorboard_callback, get_optimizer, setup_wandb)
from config import CONFIG
from constants import MODEL_CKPT_FILENAME, REPO_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

# Get the current run.
run = Run.get_context()

# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")
logger.info('DATA_DIR: %s', DATA_DIR)

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if run.id.startswith("OfflineRun"):
    logger.info('Running in offline mode...')

    # Access workspace.
    logger.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    dataset_name = CONFIG.DATASET_NAME_LOCAL
    dataset_path = get_dataset_path(DATA_DIR / 'datasets', dataset_name)
    download_dataset(workspace, dataset_name, dataset_path)

# Online run. Use dataset provided by training notebook.
else:
    logger.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace

    dataset_name = CONFIG.DATASET_NAME

    # Mount or download
    dataset_path = run.input_datasets['cgm_dataset']

# Get the QR-code paths.
dataset_path = os.path.join(dataset_path, "scans")
logger.info('Dataset path: %s', dataset_path)
#logger.info(glob.glob(os.path.join(dataset_path, "*"))) # Debug
logger.info('Getting QR-code paths...')
person_paths = glob.glob(os.path.join(dataset_path, "*"))
logger.info('person_paths: %d', len(person_paths))
assert len(person_paths) != 0

person_paths = filter_blacklisted_persons(person_paths)

# Shuffle and split into train and validate.
random.shuffle(person_paths)
split_index = int(len(person_paths) * 0.8)
person_paths_training = person_paths[:split_index]
person_paths_validate = person_paths[split_index:]

del person_paths

# Show split.
logger.info('Paths for training: \n\t' + '\n\t'.join(person_paths_training))
logger.info('Paths for validation: \n\t' + '\n\t'.join(person_paths_validate))

logger.info('Nbr of person_paths for training: %d', len(person_paths_training))
logger.info('Nbr of person_paths for validation: %d', len(person_paths_validate))

assert len(person_paths_training) > 0 and len(person_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        for code in CONFIG.CODES:
            pickle_paths.extend(glob.glob(os.path.join(path, code, "*.p")))
    return pickle_paths


# Get the pointclouds.
logger.info('Getting depthmap paths...')
paths_training = get_depthmap_files(person_paths_training)
paths_validate = get_depthmap_files(person_paths_validate)

del person_paths_training
del person_paths_validate

logger.info('Using %d files for training.', len(paths_training))
logger.info('Using %d files for validation.', len(paths_validate))

TARGET_INDEXES = getattr(CONFIG, 'TARGET_INDEXES', None)
TARGET_NAMES = getattr(CONFIG, 'TARGET_NAMES', None)
assert (TARGET_INDEXES is not None) != (TARGET_NAMES is not None), (TARGET_INDEXES, TARGET_NAMES)  # xor
LENGTH_TARGETS = len(TARGET_INDEXES) if TARGET_INDEXES else len(TARGET_NAMES)


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        if depthmap.shape[:2] != (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH):
            depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        targets = preprocess_targets(targets, TARGET_INDEXES, TARGET_NAMES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape(LENGTH_TARGETS)
    return depthmap, targets


# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset
del dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset
del dataset

# Note: Now the datasets are prepared.


def create_and_fit_model():
    # Create the model.
    input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
    model = create_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)
    model.summary()

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

    optimizer = get_optimizer(CONFIG.USE_ONE_CYCLE,
                              lr=CONFIG.LEARNING_RATE,
                              n_steps=len(paths_training) / CONFIG.BATCH_SIZE)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    # Train the model.
    model.fit(
        dataset_training.batch(CONFIG.BATCH_SIZE),
        validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
        epochs=CONFIG.EPOCHS,
        callbacks=training_callbacks,
        verbose=2
    )


if CONFIG.USE_MULTIGPU:
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Number of devices: %s", strategy.num_replicas_in_sync)
    with strategy.scope():
        create_and_fit_model()
else:
    create_and_fit_model()

# Done.
run.complete()
