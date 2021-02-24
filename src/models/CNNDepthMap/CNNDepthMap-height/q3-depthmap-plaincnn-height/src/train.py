from pathlib import Path
import os
import pickle
import random
import shutil
import sys

import glob2 as glob
import tensorflow as tf
import numpy as np
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import wandb
from wandb.keras import WandbCallback
import evidential_deep_learning as edl

from config import CONFIG, DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, MODEL_CKPT_FILENAME, REPO_DIR

# Get the current run.
run = Run.get_context()

if run.id.startswith("OfflineRun"):
    utils_dir_path = REPO_DIR / "src/common/model_utils"
    utils_paths = glob.glob(os.path.join(utils_dir_path, "*.py"))
    temp_model_util_dir = Path(__file__).parent / "tmp_model_util"
    # Remove old temp_path
    if os.path.exists(temp_model_util_dir):
        shutil.rmtree(temp_model_util_dir)
    # Copy
    os.mkdir(temp_model_util_dir)
    os.system(f'touch {temp_model_util_dir}/__init__.py')
    for p in utils_paths:
        shutil.copy(p, temp_model_util_dir)

from model import create_cnn  # noqa: E402
from tmp_model_util.preprocessing import preprocess_depthmap, preprocess_targets  # noqa: E402
from tmp_model_util.utils import calculate_mean_and_std_targets, download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback, get_optimizer, setup_wandb  # noqa: E402

# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")
print(f"DATA_DIR: {DATA_DIR}")

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if run.id.startswith("OfflineRun"):
    print("Running in offline mode...")

    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    dataset_name = CONFIG.DATASET_NAME_LOCAL
    dataset_path = get_dataset_path(DATA_DIR, dataset_name)
    download_dataset(workspace, dataset_name, dataset_path)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace

    dataset_name = CONFIG.DATASET_NAME

    # Mount or download
    if CONFIG.DATASET_MODE == DATASET_MODE_MOUNT:
        dataset_path = sys.argv[1]  # This expects the dataset_path to be the first argument to this script
    elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
        print('Run', run)
        print("input_datasets", run.input_datasets)
        print("sys.argv[1]", sys.argv[1])
        dataset_path = sys.argv[1]
        # dataset_path = run.input_datasets['input_1']

        # dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        # download_dataset(workspace, dataset_name, dataset_path)
    else:
        raise NameError(f"Unknown DATASET_MODE: {CONFIG.DATASET_MODE}")

# Get the QR-code paths.
dataset_path = os.path.join(dataset_path, "scans")
print("Dataset path:", dataset_path)
#print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
print("Getting QR-code paths...")
qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
print("qrcode_paths: ", len(qrcode_paths))
assert len(qrcode_paths) != 0

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]

del qrcode_paths

# Show split.
print("Paths for training:")
print("\t" + "\n\t".join(qrcode_paths_training))
print("Paths for validation:")
print("\t" + "\n\t".join(qrcode_paths_validate))

print(len(qrcode_paths_training))
print(len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
print("Getting depthmap paths...")
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)

del qrcode_paths_training
del qrcode_paths_validate

print("Using {} files for training.".format(len(paths_training)))
print("Using {} files for validation.".format(len(paths_validate)))

DEPTHMAP_MEAN = 0.18
DEPTHMAP_STD = 0.07

TARGET_MEAN = 91.0
TARGET_MINIMUM = 40.0
TARGET_STD = 9.7

# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)

        assert_pixel_value_min = tf.Assert(tf.reduce_min(depthmap) >= 0, [path, tf.reduce_min(depthmap), depthmap])
        assert_pixel_value_max = tf.Assert(tf.reduce_max(depthmap) < 10, [path, tf.reduce_max(depthmap), depthmap]) # 10
        with tf.control_dependencies([assert_pixel_value_min, assert_pixel_value_max]):
            depthmap = depthmap / max_value
            depthmap = (depthmap - DEPTHMAP_MEAN) / DEPTHMAP_STD
        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))

        targets = preprocess_targets(targets, CONFIG.TARGET_INDEXES)

        assert_child_height = tf.Assert(40 < targets < 150, [path, targets])  # Children should be between 40cm and 150cm
        with tf.control_dependencies([assert_child_height]):
            targets = (targets - TARGET_MINIMUM) / TARGET_STD
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets


def tf_flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset_norm
del dataset_norm

# mean, std, minimum, maximum = calculate_mean_and_std_targets(dataset_training, 20000)

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

# Note: Now the datasets are prepared.

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

optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE, clipnorm=0.5)

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=CONFIG.EDL_COEFF)


# last layer produces: mu, v, alpha, beta

def my_metric_mu(y_true, y_pred):
    return y_pred[0]
def my_metric_v(y_true, y_pred):
    return y_pred[1]
def my_metric_alpha(y_true, y_pred):
    return y_pred[2]
def my_metric_beta(y_true, y_pred):
    return y_pred[3]

def my_mae(y_true, y_pred):
    y_true_ = (y_true * TARGET_STD) + TARGET_MINIMUM
    y_pred_ = (y_pred * TARGET_STD) + TARGET_MINIMUM
    return abs(y_true_ - y_pred_)

# Compile the model.
model.compile(
    optimizer=optimizer,
    loss=EvidentialRegressionLoss,
    metrics=['mae', my_mae, my_metric_mu, my_metric_v, my_metric_alpha, my_metric_beta]
)

# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_batches,
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)

# Done.
run.complete()
