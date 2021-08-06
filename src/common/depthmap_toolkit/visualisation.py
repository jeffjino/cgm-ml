import logging
import logging.config

import numpy as np

from common.depthmap_toolkit.constants import MASK_CHILD
from common.depthmap_toolkit.depthmap import Depthmap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

CHILD_HEAD_HEIGHT_IN_METERS = 0.25
PATTERN_LENGTH_IN_METERS = 0.1
IDX_RED = 0
IDX_GREEN = 1
IDX_BLUE = 2


def blur_face(data: np.ndarray, highest_point: np.ndarray, dmap: Depthmap) -> np.ndarray:
    """Faceblur of the detected standing child.

    It uses the highest point of the child and blur all pixels in distance less than CHILD_HEAD_HEIGHT_IN_METERS.

    Args:
        data: existing canvas to blur
        highest_point: 3D point. The surroundings of this point will be blurred.
        dmap: depthmap

    Returns:
        Canvas like data with face blurred.
    """
    output = np.copy(data)
    points_3d_arr = dmap.convert_2d_to_3d_oriented()

    # blur RGB data around face
    for x in range(dmap.width):
        for y in range(dmap.height):

            # count distance from the highest child point
            depth = dmap.depthmap_arr[x, y]
            if not depth:
                continue
            point = points_3d_arr[:, x, y]

            vector = point - highest_point
            distance = abs(vector[0]) + abs(vector[1]) + abs(vector[2])
            if distance >= CHILD_HEAD_HEIGHT_IN_METERS:
                continue

            # Gausian blur
            pixel = np.array([0, 0, 0])
            count = 0
            step = 5
            for tx in range(x - step, x + step):
                for ty in range(y - step, y + step):
                    if not (0 < tx < dmap.width and 0 < ty < dmap.height):
                        continue
                    index = dmap.height - ty - 1
                    pixel = pixel + data[tx, index, 0]
                    count = count + 1
            index = dmap.height - y - 1
            output[x, index] = pixel / count

    return output


def render_confidence(dmap: Depthmap):
    confidence = dmap.confidence_arr
    confidence[confidence == 0.] = 1.

    confidence = np.fliplr(confidence)  # flip left-right
    output = np.stack([confidence, confidence, confidence], axis=2)
    return output


def render_depth(dmap: Depthmap, use_smooth=False) -> np.ndarray:
    """Render depthmap into a 2D image.

    We assume here that all values in dmap.depthmap_arr are positive.

    A distance of 0m is visualized in white.
    A distance of 2m is visualized in black.

    flashlight analogy: close-to-cam data is white
    """
    if use_smooth:
        dmap_arr = np.minimum(dmap.depthmap_arr_smooth / 2., 1.)
    else:
        dmap_arr = np.minimum(dmap.depthmap_arr / 2., 1.)

    cond = (dmap_arr != 0.)
    dmap_arr[cond] = 1. - dmap_arr[cond]

    dmap_arr = np.fliplr(dmap_arr)  # flip left-right
    output = np.stack([dmap_arr, dmap_arr, dmap_arr], axis=2)
    return output


def render_normal(dmap: Depthmap) -> np.ndarray:
    """Render normal vectors

    How normal vector are visualized:
    When a vector has (x,y,z)=(1,0,0), this will show in red color.
    When a vector has (x,y,z)=(0,1,0), this will show in green color (e.g. floor).
    When a vector has (x,y,z)=(0,0,1), this will show in blue color.
    """

    points_3d_arr = dmap.convert_2d_to_3d_oriented(should_smooth=True)
    normal = dmap.calculate_normalmap_array(points_3d_arr)

    # We can't see negative values, so we take the absolute value
    normal = abs(normal)  # 3 x w x h

    output = np.moveaxis(normal, 0, -1)
    output = np.fliplr(output)  # flip left-right
    return output


def render_rgb(dmap: Depthmap) -> np.ndarray:
    output = np.copy(dmap.rgb_array)  # shape (height, width, 3)
    output = output / 255.
    output = np.moveaxis(output, 0, 1)
    output = np.fliplr(output)  # flip left-right
    return output


def render_segmentation(floor: float,
                        mask: np.ndarray,
                        dmap: Depthmap) -> np.ndarray:
    output = np.zeros((dmap.width, dmap.height, 3))

    point = dmap.convert_2d_to_3d_oriented(should_smooth=True)
    normal = dmap.calculate_normalmap_array(point)

    for x in range(dmap.width):
        for y in range(dmap.height):

            # get depth value
            depth = dmap.depthmap_arr[x, y]
            if not depth:
                continue

            # segmentation visualisation
            horizontal = (point[1, x, y] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical_x = (point[0, x, y] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical_z = (point[2, x, y] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical = (vertical_x + vertical_z) / 2.0
            index = dmap.height - y - 1

            if mask[x, y] == MASK_CHILD:
                # Red + Green = Yellow
                output[x, index, IDX_RED] = horizontal
                output[x, index, IDX_GREEN] = horizontal
            elif abs(normal[1, x, y]) < 0.5:
                output[x, index, IDX_RED] = horizontal
            else:
                if abs(point[1, x, y] - floor) < 0.1:
                    output[x, index, IDX_BLUE] = vertical
                else:
                    output[x, index, IDX_GREEN] = vertical

    # Fog effect: achieved by deviding by depth*depth
    fog_normalization = np.expand_dims(dmap.depthmap_arr * dmap.depthmap_arr, axis=-1)
    fog_normalization[fog_normalization == 0.] = 1.
    output = output / fog_normalization

    # Ensure pixel clipping
    np.clip(output, 0., 1., output)

    return output


def render_plot(dmap: Depthmap) -> np.ndarray:
    # detect floor and child
    floor: float = dmap.get_floor_level()
    mask = dmap.segment_child(floor)  # dmap.detect_floor(floor)
    highest_point: np.ndarray = dmap.get_highest_point(mask)
    logger.info('height=%fm', highest_point[1] - floor)

    # prepare plots
    output_plots = [
        render_depth(dmap),
        render_normal(dmap),
        render_segmentation(floor, mask, dmap),
        render_confidence(dmap),
    ]
    if dmap.has_rgb:
        output_rgb = render_rgb(dmap)
        output_rgb = blur_face(output_rgb, highest_point, dmap)
        output_plots.append(output_rgb)

    output = np.concatenate(output_plots, axis=1)
    return output
