import os
import random
from pathlib import Path
import logging

import glob2 as glob
import tensorflow as tf
import tensorflow_datasets as tfds
from tmp_model_util.utils import download_dataset, get_dataset_path

from config import DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, REPO_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


def create_datasets(workspace, experiment, run, offline_run, config):

    data_dir = REPO_DIR / 'data' if offline_run else Path(".")
    logger.info('data_dir: %s', data_dir)

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if offline_run:
        dataset_name = config.DATASET_NAME_LOCAL
        dataset_path = get_dataset_path(data_dir, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)

    # Online run. Use dataset provided by training notebook.
    else:
        dataset_name = config.DATASET_NAME

        # Mount or download
        if config.DATASET_MODE == DATASET_MODE_MOUNT:
            dataset_path = run.input_datasets["dataset"]
        elif config.DATASET_MODE == DATASET_MODE_DOWNLOAD:
            dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
            download_dataset(workspace, dataset_name, dataset_path)
        else:
            raise NameError(f"Unknown DATASET_MODE: {config.DATASET_MODE}")

    logger.info('Using dataset path: %s', dataset_path)

    # Branch into specific datasets.
    if dataset_name == "anon_rgb_training":
        return __create_anon_rgb_training(dataset_path, config)
    elif dataset_name == "anomaly_detection_data":
        return __create_anomaly_detection_data(dataset_path, config)
    else:
        assert False, f"Unexpected dataset {dataset_name}."


def __create_anon_rgb_training(dataset_path, config):

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    logger.info('Dataset path: %s', dataset_path)
    #logger.info(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    logger.info('Getting QR-code paths...')
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    logger.info('qrcode_paths: %d', len(qrcode_paths))
    assert len(qrcode_paths) != 0

    # Restricting to a fixed number of scans.
    logger.info("Slicing scans...")
    qrcode_paths = qrcode_paths[:config.DATASET_MAX_SCANS]
    logger.info('qrcode_paths: %d', len(qrcode_paths))
    #assert len(qrcode_paths) == config.DATASET_MAX_SCANS

    # Shuffle and split into train and validate.
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

    # For mapping scan paths to samples.
    # Files are shuffled per scan.
    # And restricted to a maximum of samples per scan.
    def get_files(scan_paths, extension="jpg"):
        result_paths = []
        for scan_path in scan_paths:
            sample_paths = glob.glob(os.path.join(scan_path, "**", "*." + extension))
            random.shuffle(sample_paths)
            sample_paths = sample_paths[:config.DATASET_MAX_SAMPLES_PER_SCAN]
            result_paths.extend(sample_paths)
        return result_paths

    # Get the samples.
    logger.info('Getting paths...')
    paths_training = get_files(qrcode_paths_training)
    paths_validate = get_files(qrcode_paths_validate)
    logger.info('Using %d files for training.', len(paths_training))
    logger.info('Using %d files for validation.', len(paths_validate))
    assert paths_training != 0
    assert paths_validate != 0

    # Shuffle all.
    random.shuffle(paths_training)
    random.shuffle(paths_validate)

    # QR codes not necessary anymore.
    del qrcode_paths_training
    del qrcode_paths_validate

    # Function for loading and processing images.
    def tf_load_sample(path):
        def py_load_sample(path):
            image_string = tf.io.read_file(path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = tf.image.resize(image, (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH))
            image = tf.image.rot90(image, k=3)
            image = image / 255.0
            return image

        image = tf.py_function(py_load_sample, [path], [tf.float32])[0]
        image.set_shape((config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH, 3))
        return image

    # Create dataset for training.
    # Note: Model will do caching et cetera.
    paths = paths_training
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: tf_load_sample(path))
    dataset_train = dataset
    del dataset

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_validate
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: tf_load_sample(path))
    dataset_validate = dataset
    del dataset

    # Create dataset for anomaly detection.
    # Note: Model will do caching et cetera.
    def tf_preprocess(image):
        image = tf.image.resize(image, (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)) / 255.0
        return image

    dataset_anomaly = tfds.load("cats_vs_dogs", split="train[:10%]")
    dataset_anomaly = dataset_anomaly.map(lambda sample: tf_preprocess(sample["image"]))

    # Done.
    return dataset_train, dataset_validate, dataset_anomaly


def __create_anomaly_detection_data(dataset_path, config):

    #dataset_path = os.path.join(dataset_path, "scans")
    logger.info(glob.glob(os.path.join(dataset_path, "*")))

    good_paths = glob.glob(os.path.join(dataset_path, "not_bad", "*.jpg"))
    bad_paths = glob.glob(os.path.join(dataset_path, "bad", "*.jpg"))

    random.shuffle(good_paths)
    split_index = int(0.8 * len(good_paths))
    paths_training = good_paths[:split_index]
    paths_validate = good_paths[split_index:]
    paths_anomaly = bad_paths

    logger.info('Training paths: %d', len(paths_training))
    logger.info('Validate paths: %d', len(paths_validate))
    logger.info('Anomaly paths: %d', len(paths_anomaly))
    assert len(paths_training) * len(paths_validate) * len(paths_anomaly) != 0

    # Function for loading and processing images.
    def tf_load_sample(path):
        def py_load_sample(path):
            image_string = tf.io.read_file(path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = tf.image.resize(image, (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH))
            image = tf.image.rot90(image, k=3)
            image = image / 255.0
            return image

        image = tf.py_function(py_load_sample, [path], [tf.float32])[0]
        image.set_shape((config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH, 3))
        return image

    # Create dataset for training.
    # Note: Model will do caching et cetera.
    paths = paths_training
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: tf_load_sample(path))
    dataset_train = dataset
    del dataset

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_validate
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: tf_load_sample(path))
    dataset_validate = dataset
    del dataset

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_anomaly
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: tf_load_sample(path))
    dataset_anomaly = dataset
    del dataset

    return dataset_train, dataset_validate, dataset_anomaly
