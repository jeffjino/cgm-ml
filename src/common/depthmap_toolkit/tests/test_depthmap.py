from pathlib import Path

import numpy as np

from common.depthmap_toolkit.depthmap import Depthmap, smoothen_depthmap_array

TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def test_depthmap():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_282.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_282.jpg'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')

    dmap = Depthmap.create_from_zip(depthmap_dir, depthmap_fname, rgb_fname, calibration_file)
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
    assert mask.shape[0] == dmap.rgb_array.shape[1]
    assert mask.shape[1] == dmap.rgb_array.shape[0]

    angle_in_degrees = dmap.get_angle_between_camera_and_floor()
    assert -90 < angle_in_degrees < 90


def test_get_highest_point():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_282.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_282.jpg'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')
    dmap = Depthmap.create_from_zip(depthmap_dir, depthmap_fname, rgb_fname, calibration_file)

    # Find top of the object
    floor = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    highest_point = dmap.get_highest_point(mask)  # 3D

    object_height_in_m = highest_point[1] - floor
    assert 0.3 < object_height_in_m < 0.6


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
