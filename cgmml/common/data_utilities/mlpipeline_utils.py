"""Preprocessing utilities

In order to preprocess ZIP file to extract a depthmap, we use this code:
https://github.com/Welthungerhilfe/cgm-rg/blob/92efa0febb91c9656ce8e5dbfad953ff7ce721a9/src/utils/preprocessing.py#L12

file of minor importance:
https://github.com/Welthungerhilfe/cgm-ml/blob/c8be9138e025845bedbe7cfc0d131ef668e01d4b/old
/cgm_database/command_preprocess.py#L92
"""

from pathlib import Path
import pickle
from typing import Tuple
import tempfile

import numpy as np
from skimage.transform import resize
from PIL import Image

from cgmml.common.depthmap_toolkit.depthmap import Depthmap, parse_calibration

TOOLKIT_DIR = Path(__file__).parents[1] / 'depthmap_toolkit'
CALIBRATION_FPATH = TOOLKIT_DIR / "camera_calibration_p30pro_EU.txt"
NORMALIZATION_VALUE = 7.5
IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH = 180, 240


class InvalidDevicePoseError(Exception):
    pass


def preprocess_depthmap(depthmap: np.ndarray) -> np.ndarray:
    return depthmap.astype("float32")


def preprocess(depthmap: np.ndarray) -> np.ndarray:
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE

    width, height = depthmap.shape
    assert width / IMAGE_TARGET_WIDTH == height / IMAGE_TARGET_HEIGHT, f"\
        {width} / {IMAGE_TARGET_WIDTH} == {height} / {IMAGE_TARGET_HEIGHT}"
    depthmap = resize(depthmap, (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT))

    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap


def preprocess_rgb(rgb_array: np.ndarray) -> np.ndarray:
    """Preprocess RGB

    Args:
        rgb_array: shape (height, width, 3)

    Returns:
        proprocessed rgb array
    """
    rgb_array = rgb_array.astype("float32")
    rgb_array = rgb_array / 255.

    width, height, _ = rgb_array.shape
    assert width / IMAGE_TARGET_WIDTH == height / IMAGE_TARGET_HEIGHT, f"\
        {width} / {IMAGE_TARGET_WIDTH} == {height} / {IMAGE_TARGET_HEIGHT}"
    rgb_array = resize(rgb_array, (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT, 3))

    return rgb_array


def create_layers(depthmap_fpath: str) -> Tuple[np.ndarray, dict]:
    dmap = Depthmap.create_from_zip_absolute(depthmap_fpath, rgb_fpath=None, calibration_fpath=str(CALIBRATION_FPATH))
    depthmap = dmap.depthmap_arr  # shape: (width, height)
    depthmap = preprocess(depthmap)
    layers = depthmap
    if not dmap.device_pose:
        raise InvalidDevicePoseError()
    metadata = {
        'device_pose': dmap.device_pose,
        'raw_header': dmap.header,
        'angle': dmap.get_angle_between_camera_and_floor(),
    }
    return layers, metadata


def rotate_and_load_depthmap_with_rgbd(depthmap_fpath: str, rgb_fpath: str, calibration_fpath: str) -> Depthmap:
    width, height, data, depth_scale, max_confidence, device_pose, header_line = (
        Depthmap.read_depthmap_data(depthmap_fpath))

    with tempfile.NamedTemporaryFile() as rgb_temp_file:
        pil_im = Image.open(rgb_fpath)
        pil_im = pil_im.rotate(90, expand=True)
        pil_im.save(rgb_temp_file.name, 'png')
        rgb_array = Depthmap.read_rgb_data(rgb_temp_file.name, width, height)

    intrinsics = parse_calibration(calibration_fpath)
    depthmap_arr = None
    rgb_fpath = None

    dmap = Depthmap(np.array(intrinsics), width, height, data, depthmap_arr,
                    depth_scale, max_confidence, device_pose,
                    rgb_fpath, rgb_array, header_line)
    return dmap


def create_layers_rgbd(depthmap_fpath: str, rgb_fpath: str, should_rotate_rgb: bool) -> Tuple[np.ndarray, dict]:
    if should_rotate_rgb:
        dmap = Depthmap.create_from_zip_absolute(depthmap_fpath, rgb_fpath, str(CALIBRATION_FPATH))
    else:
        dmap = rotate_and_load_depthmap_with_rgbd(depthmap_fpath, rgb_fpath, str(CALIBRATION_FPATH))

    if not dmap.device_pose:
        raise InvalidDevicePoseError()

    depthmap = dmap.depthmap_arr  # shape: (longer, shorter)
    depthmap = preprocess(depthmap)  # shape (longer, shorter, 1)

    rgb = dmap.rgb_array  # shape (longer, shorter, 3)
    rgb = preprocess_rgb(rgb)  # shape (longer, shorter, 3)

    layers = np.concatenate([
        depthmap,  # shape (longer, shorter, 1)
        rgb,  # shape (longer, shorter, 3)
    ], axis=2)  # shape (longer, shorter, 4)

    metadata = {
        'device_pose': dmap.device_pose,
        'raw_header': dmap.header,
        'angle': dmap.get_angle_between_camera_and_floor(),
    }
    return layers, metadata


class ArtifactProcessor:
    def __init__(self, input_dir: str, output_dir: str, dataset_type: str, should_rotate_rgb: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        assert dataset_type in ['depthmap', 'rgbd']
        self.dataset_type = dataset_type
        self.should_rotate_rgb = should_rotate_rgb

    def create_and_save_pickle(self, artifact_dict: dict) -> str:
        """Side effect: Saves and returns file path"""
        # Prepare data to save
        zip_input_full_path = f"{self.input_dir}/{artifact_dict['file_path']}"

        try:
            if self.dataset_type == 'depthmap':
                layers, metadata = create_layers(zip_input_full_path)
            elif self.dataset_type == 'rgbd':
                rgb_input_full_path = f"{self.input_dir}/{artifact_dict['file_path_rgb']}"
                layers, metadata = create_layers_rgbd(zip_input_full_path, rgb_input_full_path, self.should_rotate_rgb)
            else:
                raise NameError(self.dataset_type)
        except InvalidDevicePoseError:
            return ''
        target_dict = {**artifact_dict, **metadata}

        # Prepare path
        timestamp = artifact_dict['timestamp']
        scan_id = artifact_dict['scan_id']
        scan_step = artifact_dict['scan_step']
        order_number = artifact_dict['order_number']
        person_id = artifact_dict['person_id']
        pickle_output_path = f"scans/{person_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"

        # Write into pickle
        pickle_output_full_path = f"{self.output_dir}/{pickle_output_path}"
        Path(pickle_output_full_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((layers, target_dict), open(pickle_output_full_path, "wb"))

        return pickle_output_full_path
