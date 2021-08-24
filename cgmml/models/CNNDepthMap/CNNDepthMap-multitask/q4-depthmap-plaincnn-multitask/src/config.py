from bunch import Bunch

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = Bunch(dict(
    DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME="anon-depthmap-95k",
    DATASET_NAME_LOCAL="anon-depthmap-mini",
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=500,
    BATCH_SIZE=256,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.0007,
    USE_ONE_CYCLE=True,
    USE_MULTIGPU=False,
    CLUSTER_NAME='gpu-cluster',

    # Parameters for dataset generation.
    TARGET_INDEXES=[0, 1],  # 0 is height, 1 is weight.
    HEIGHT_IMPORTANCE=0.2,
    WEIGHT_IMPORTANCE=0.8,
))
