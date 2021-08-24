from pathlib import Path

import numpy as np

from cgmml.common.depthmap_toolkit.depthmap import Depthmap, smoothen_depthmap_array
from cgmml.common.depthmap_toolkit.constants import MASK_CHILD

TOOLKIT_DIR = Path(__file__).parents[0].absolute()
DEPTHMAP_DIR = TOOLKIT_DIR / 'huawei_p40pro'
DEPTHMAP_FPATH = DEPTHMAP_DIR / 'depth' / 'depth_dog_1622182020448_100_282.depth'
RGB_FPATH = DEPTHMAP_DIR / 'rgb' / 'rgb_dog_1622182020448_100_282.jpg'
CALIBRATION_FILE = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')


def test_depthmap():
    dmap = Depthmap.create_from_zip_absolute(DEPTHMAP_FPATH, RGB_FPATH, CALIBRATION_FILE)
    assert dmap.width == 240
    assert dmap.height == 180

    expected_intrinsics = np.array([
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
    ])
    np.testing.assert_array_almost_equal(dmap.intrinsics, expected_intrinsics)
    assert dmap.max_confidence == 7.
    assert dmap.depth_scale == 0.001

    floor = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    highest_point = dmap.get_highest_point(mask)
    child_height_in_m = highest_point[1] - floor
    assert 0 < child_height_in_m < 1.2
    assert mask.shape[:2] == dmap.rgb_array.shape[:2]

    angle_in_degrees = dmap.get_angle_between_camera_and_floor()
    assert -90 < angle_in_degrees < 90

    distance_in_m = dmap.get_distance_of_child_from_camera(mask)
    assert 0.1 < distance_in_m < 5.


def test_get_highest_point():
    dmap = Depthmap.create_from_zip_absolute(DEPTHMAP_FPATH, RGB_FPATH, CALIBRATION_FILE)

    # Find top of the object
    floor = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    highest_point = dmap.get_highest_point(mask)  # 3D

    object_height_in_m = highest_point[1] - floor
    assert 0.3 < object_height_in_m < 0.6


def test_is_child_fully_visible():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_282.depth'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')
    dmap = Depthmap.create_from_zip(depthmap_dir, depthmap_fname, 0, calibration_file)

    # Run standard normal visibility check
    floor = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    assert dmap.is_child_fully_visible(mask)

    # Run visibility check when child is covering most of the camera
    margin = 5
    x1 = margin
    x2 = dmap.width - margin
    y1 = margin
    y2 = dmap.height - margin
    mask[x1:x2, y1:y2] = MASK_CHILD
    assert not dmap.is_child_fully_visible(mask)


def test_smoothen_depthmap_array_no_masking():
    depthmap = np.array([
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
    ])
    np.testing.assert_array_equal(depthmap, smoothen_depthmap_array(depthmap))


def test_smoothen_depthmap_array_with_masking():
    depthmap = np.array([
        [1., 1., 1., 1.],
        [0., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
    ])
    expected = np.array([
        [0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 1., 1., 1.],
        [1., 1., 1., 1.],
    ])
    np.testing.assert_array_equal(expected, smoothen_depthmap_array(depthmap))
