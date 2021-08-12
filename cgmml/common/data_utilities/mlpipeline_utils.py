"""Preprocessing utilities

In order to preprocess ZIP file to extract a depthmap, we use this code:
https://github.com/Welthungerhilfe/cgm-rg/blob/92efa0febb91c9656ce8e5dbfad953ff7ce721a9/src/utils/preprocessing.py#L12

file of minor importance:
https://github.com/Welthungerhilfe/cgm-ml/blob/c8be9138e025845bedbe7cfc0d131ef668e01d4b/old
/cgm_database/command_preprocess.py#L92
"""

from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np
from skimage.transform import resize

from cgmml.common.depthmap_toolkit.depthmap import Depthmap

TOOLKIT_DIR = Path(__file__).parents[1] / 'depthmap_toolkit'
NORMALIZATION_VALUE = 7.5
IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH = 180, 240


def preprocess_depthmap(depthmap: np.ndarray) -> np.ndarray:
    return depthmap.astype("float32")


def preprocess(depthmap: np.ndarray) -> np.ndarray:
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = resize(depthmap, (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap


def create_layers(depthmap_fpath: str) -> Tuple[np.ndarray, dict]:
    calibration_fpath = TOOLKIT_DIR / "camera_calibration_p30pro_EU.txt"

    dmap = Depthmap.create_from_zip_absolute(depthmap_fpath,
                                             rgb_fpath=None,
                                             calibration_fpath=calibration_fpath,
                                             )
    depthmap = dmap.depthmap_arr  # shape: (width, height)
    depthmap = preprocess(depthmap)

    layers = np.concatenate([
        depthmap,
    ])

    metadata = {
        'device_pose': dmap.device_pose,
        'raw_header': dmap.header,
    }

    return layers, metadata


def create_layers_from_multiple_paths(fpaths: List[str]) -> np.ndarray:
    depthmaps = []
    for fpath in fpaths:
        depthmap = create_layers(fpath)
        depthmaps.append(depthmap)
    depthmaps = np.array(depthmaps)
    return depthmaps


class ArtifactProcessor:
    def __init__(self, input_dir: str, output_dir: str, idx2col: Dict[int, str]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.idx2col = idx2col

    def create_and_save_pickle(self, zip_input_full_path: str, artifact_dict: dict) -> str:
        """Side effect: Saves and returns file path"""
        # Prepare data to save
        layers, metadata = create_layers(zip_input_full_path)
        target_dict = {**artifact_dict, **metadata}

        # Prepare path
        timestamp = artifact_dict['timestamp']
        scan_id = artifact_dict['scan_id']
        scan_step = artifact_dict['scan_step']
        order_number = artifact_dict['order_number']
        pickle_output_path = f"scans/{scan_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"

        # Write into pickle
        pickle_output_full_path = f"{self.output_dir}/{pickle_output_path}"
        Path(pickle_output_full_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((layers, target_dict), open(pickle_output_full_path, "wb"))

        return pickle_output_full_path

    def process_artifact_tuple(self, artifact_tuple: tuple):
        """Side effect: Saves and returns file path"""
        artifact_dict = {self.idx2col[i]: el for i, el in enumerate(artifact_tuple)}
        zip_input_full_path = f"{self.input_dir}/{artifact_dict['file_path']}"
        pickle_output_full_path = self.create_and_save_pickle(zip_input_full_path, artifact_dict)
        return pickle_output_full_path
