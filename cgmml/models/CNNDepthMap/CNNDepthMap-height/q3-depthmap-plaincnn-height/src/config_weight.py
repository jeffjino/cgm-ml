from bunch import Bunch

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = Bunch(dict(
    DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME="depthmap-19k",
    DATASET_NAME_LOCAL="depthmap-19k",
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=1000,
    BATCH_SIZE=256,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.0003,
    USE_ONE_CYCLE=True,
    USE_DROPOUT=False,
    USE_WANDB=False,
    USE_MULTIGPU=False,
    CLUSTER_NAME='gpu-cluster-V100',

    # Parameters for dataset generation.
    TARGET_NAMES=['weight'],
    CODES=['100', '101', '102', '200', '201', '202'],
))
