import datetime
import os
from pathlib import Path
import subprocess

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import models, layers
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from tensorflow.keras import callbacks


def create_base_cnn(input_shape, dropout):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.05))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.075))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.125))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.175))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.25))

    return model


def create_head(input_shape, dropout, name="head"):
    model = models.Sequential(name=name)
    model.add(layers.Dense(128, activation="relu", input_shape=input_shape))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(16, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation="linear"))
    return model


def get_optimizer(use_one_cycle: bool, lr: float, n_steps: int) -> tf.python.keras.optimizer_v2.optimizer_v2.OptimizerV2:
    if use_one_cycle:
        lr_schedule = tfa.optimizers.TriangularCyclicalLearningRate(
            initial_learning_rate=lr / 100,
            maximal_learning_rate=lr,
            step_size=n_steps,
        )
        # Note: When using 1cycle, this uses the Adam (not Nadam) optimizer
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return tf.keras.optimizers.Nadam(learning_rate=lr)


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    print("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    print(f"Downloading dataset {dataset_name}.. Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset.download(target_path=dataset_path, overwrite=False)
    print(f"Finished downloading {dataset_name}, Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_dataset_path(data_dir: Path, dataset_name: str) -> str:
    return str(data_dir / dataset_name)


class AzureLogCallback(callbacks.Callback):
    """Pushes metrics and losses into the run on AzureML"""
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                self.run.log(key, value)


def create_tensorboard_callback() -> callbacks.TensorBoard:
    return callbacks.TensorBoard(
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


WANDB_API_KEY_MH = "237ca046c5dcd915945761dc477207549ef2c42c"


def setup_wandb():
    wandb_login = subprocess.run(["wandb", "login", WANDB_API_KEY_MH])
    assert wandb_login.returncode == 0



import numpy as np

def calculate_mean_and_std(dataset, NUM_ARTIFACTS=5):
    aaa = dataset.take(NUM_ARTIFACTS)
    ccc = list(aaa.as_numpy_iterator())

    # ccc[0][0].shape == (240, 180, 1)
    # ccc[1][0].shape == (1,)

    depthmaps = []
    for item in ccc:
        depthmap = item[0]
        # print(depthmap.min(), depthmap.max())
        depthmaps.append(depthmap)

    depthmaps_np = np.array(depthmaps)  # shape (5, 240, 180, 1)

    std = depthmaps_np.std()
    mean = depthmaps_np.mean()
    return mean, std

def calculate_mean_and_std_targets(dataset, NUM_ARTIFACTS=5):
    aaa = dataset.take(NUM_ARTIFACTS)
    ccc = list(aaa.as_numpy_iterator())

    # ccc[0][0].shape == (240, 180, 1)
    # ccc[1][0].shape == (1,)

    targets = []
    for item in ccc:
        target = item[1]
        targets.append(target)

    targets_np = np.array(targets)

    std = targets_np.std()
    mean = targets_np.mean()
    minimum = targets_np.min()
    maximum = targets_np.max()
    return mean, std, minimum, maximum

# mean, std = calculate_mean_and_std(dataset_training, 20000)
# mean, std = calculate_mean_and_std_targets(dataset_training, 20000)
# x_new = (x - mean) / std