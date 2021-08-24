from pathlib import Path
import os
import random
import logging

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import callbacks, layers, models

from cgmml.common.model_utils.preprocessing import filter_blacklisted_qrcodes
from cgmml.common.model_utils.preprocessing_multiartifact_python import (
    create_multiartifact_paths_for_qrcodes)
from cgmml.common.model_utils.preprocessing_multiartifact_tensorflow import (
    create_multiartifact_sample)
from cgmml.common.model_utils.utils import (
    download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback, get_optimizer)
from cgmml.common.model_utils.model_plaincnn import create_head
from model import get_base_model  # model.py relies on temp_common.model_utils
from config import CONFIG
from constants import MODEL_CKPT_FILENAME, REPO_DIR
from augmentation import tf_augment_sample

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

    logger.info('Accessing dataset...')
    dataset_name = CONFIG.DATASET_NAME_LOCAL
    dataset_path = get_dataset_path(DATA_DIR, dataset_name)
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
dataset_scans_path = os.path.join(dataset_path, "scans")
logger.info('Dataset path: %s', dataset_scans_path)
#logger.info(glob.glob(os.path.join(dataset_scans_path, "*"))) # Debug
logger.info('Getting QR-code paths...')
qrcode_paths = glob.glob(os.path.join(dataset_scans_path, "*"))
logger.info('qrcode_paths: %d', len(qrcode_paths))
assert len(qrcode_paths) != 0

qrcode_paths = filter_blacklisted_qrcodes(qrcode_paths)

# Shuffle and split into train and validate.
random.seed(CONFIG.SPLIT_SEED)
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]

del qrcode_paths

# Show split.
logger.info('Paths for training: \n\t' + '\n\t'.join(qrcode_paths_training))
logger.info('Paths for validation: \n\t' + '\n\t'.join(qrcode_paths_validate))

logger.info('Nbr of qrcode_paths for training: %d', len(qrcode_paths_training))
logger.info('Nbr of qrcode_paths for validation: %d', len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0

paths_training = create_multiartifact_paths_for_qrcodes(qrcode_paths_training, CONFIG)
logger.info('Using %d files for training.', len(paths_training))

paths_validate = create_multiartifact_paths_for_qrcodes(qrcode_paths_validate, CONFIG)
logger.info('Using %d files for validation.', len(paths_validate))


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def tf_load_pickle(path):
    """Load and process depthmaps"""
    params = [path,
              CONFIG.NORMALIZATION_VALUE,
              CONFIG.IMAGE_TARGET_HEIGHT,
              CONFIG.IMAGE_TARGET_WIDTH,
              tf.constant(CONFIG.TARGET_INDEXES),
              CONFIG.N_ARTIFACTS]
    depthmap, targets = tf.py_function(create_multiartifact_sample, params, [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets  # (240,180,5), (1,)


# Create dataset for training.
paths = paths_training  # list
dataset = tf.data.Dataset.from_tensor_slices(paths)  # TensorSliceDataset  # List[ndarray[str]]
dataset = dataset.cache()
dataset = dataset.repeat(CONFIG.N_REPEAT_DATASET)

dataset = dataset.map(tf_load_pickle, tf.data.experimental.AUTOTUNE)  # (240,180,5), (1,)

dataset = dataset.map(tf_augment_sample, tf.data.experimental.AUTOTUNE)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(tf_load_pickle, tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset
del dataset

# Note: Now the datasets are prepared.


def create_and_fit_model():
    # Create the base model
    base_model = get_base_model(workspace, DATA_DIR)
    base_model.summary()
    assert base_model.output_shape == (None, 128)

    # Create the head
    head_input_shape = (128 * CONFIG.N_ARTIFACTS,)
    head_model = create_head(head_input_shape, dropout=CONFIG.USE_DROPOUT)

    # Implement artifact flow through the same model
    model_input = layers.Input(
        shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS)
    )

    features_list = []
    for i in range(CONFIG.N_ARTIFACTS):
        features_part = model_input[:, :, :, i:i + 1]
        features_part = base_model(features_part)
        features_list.append(features_part)

    concatenation = tf.keras.layers.concatenate(features_list, axis=-1)
    assert concatenation.shape.as_list() == tf.TensorShape((None, 128 * CONFIG.N_ARTIFACTS)).as_list()
    model_output = head_model(concatenation)

    model = models.Model(model_input, model_output)
    model.summary()

    best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    training_callbacks = [
        AzureLogCallback(run),
        create_tensorboard_callback(),
        checkpoint_callback,
    ]

    optimizer = get_optimizer(CONFIG.USE_ONE_CYCLE,
                              lr=CONFIG.LEARNING_RATE,
                              n_steps=len(paths_training) / CONFIG.BATCH_SIZE)

    # Compile the model.
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Train the model.
    model.fit(
        dataset_training.batch(CONFIG.BATCH_SIZE),
        validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
        epochs=CONFIG.EPOCHS,
        callbacks=training_callbacks,
        verbose=2
    )

    if CONFIG.EPOCHS_TUNE:
        # Un-freeze
        for layer in base_model._layers:
            layer.trainable = True

        # Adjust learning rate
        optimizer = tf.keras.optimizers.Nadam(learning_rate=CONFIG.LEARNING_RATE_TUNE)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        logger.info('Start fine-tuning')
        model.fit(
            dataset_training.batch(CONFIG.BATCH_SIZE),
            validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
            epochs=CONFIG.EPOCHS_TUNE,
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
