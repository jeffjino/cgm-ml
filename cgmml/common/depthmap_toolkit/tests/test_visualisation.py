from pathlib import Path

import pytest
import numpy as np

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.visualisation import blur_face, render_rgb

TOOLKIT_DIR = Path(__file__).parents[0].absolute()
OFFSET_X_Y = (0.37, 0.53)
DEPTHMAP_DIR = TOOLKIT_DIR / 'huawei_p40pro'
DEPTHMAP_FPATH = DEPTHMAP_DIR / 'depth' / 'depth_dog_1622182020448_100_282.depth'
RGB_FPATH = DEPTHMAP_DIR / 'rgb' / 'rgb_dog_1622182020448_100_282.jpg'
CALIBRATION_FPATH = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')


def test_blur_face():
    dmap = Depthmap.create_from_zip_absolute(DEPTHMAP_FPATH, RGB_FPATH, CALIBRATION_FPATH)

    # Find top of the object
    floor = dmap.get_floor_level()
    assert floor == pytest.approx(-0.9706086, 0.001)
    mask = dmap.segment_child(floor)
    highest_point = dmap.get_highest_point(mask)  # 3D

    # Render the color data
    output_unblurred = render_rgb(dmap)

    # Blur
    output_blurred = blur_face(output_unblurred, highest_point, dmap)

    # Assert some pixels in whole image change (image not same)
    all_count = dmap.width * dmap.height
    count = np.count_nonzero(output_unblurred - output_blurred) / 3
    ratio_blurred = count / all_count
    assert 0.01 < ratio_blurred < 0.9

    # Assert that blurred around object
    object_x = int(dmap.width * OFFSET_X_Y[0])
    object_y = int(dmap.height * (1.0 - OFFSET_X_Y[1]))
    slice_x = slice(object_x - 2, object_x + 2)
    slice_y = slice(object_y - 2, object_y + 2)
    assert (output_unblurred[slice_x, slice_y] != output_blurred[slice_x, slice_y]).any()

    # Assert that NOT blurred around corner
    corner_x = 0
    corner_y = 0
    slice_x = slice(corner_x, corner_x + 4)
    slice_y = slice(corner_y, corner_y + 4)
    np.testing.assert_array_equal(output_unblurred[slice_x, slice_y], output_blurred[slice_x, slice_y])
