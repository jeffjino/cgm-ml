from collections import namedtuple
import zipfile
import math
import sys
from typing import List, Optional, Tuple
from pathlib import Path

from scipy import ndimage
import numpy as np
from PIL import Image

from common.depthmap_toolkit.depthmap_utils import matrix_calculate, IDENTITY_MATRIX_4D, parse_numbers
from common.depthmap_toolkit.constants import EXTRACTED_DEPTH_FILE_NAME, MASK_FLOOR, MASK_CHILD, MASK_INVALID

TOOLKIT_DIR = Path(__file__).parents[0].absolute()


Segment = namedtuple('Segment', 'id aabb')


def extract_depthmap(depthmap_dir: str, depthmap_fname: str):
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(depthmap_dir) / 'depth' / depthmap_fname, 'r') as zip_ref:
        zip_ref.extractall(TOOLKIT_DIR)
    return TOOLKIT_DIR / EXTRACTED_DEPTH_FILE_NAME


def smoothen_depthmap_array(image_arr: np.ndarray) -> np.ndarray:
    """Smoothen image array by averaging with direct neighbor pixels.

    Args:
        image_arr: shape (width, height)

    Returns:
        shape (width, height)
    """

    # Apply a convolution
    conv_filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    conv_filter = conv_filter / conv_filter.sum()
    smooth_image_arr = ndimage.convolve(image_arr, conv_filter)

    # Create mask which is 0 if any of the pixels in the convolution is 0
    smooth_center = (image_arr == 0.)
    smooth_right = np.zeros(image_arr.shape, dtype=bool)
    smooth_left = np.zeros(image_arr.shape, dtype=bool)
    smooth_up = np.zeros(image_arr.shape, dtype=bool)
    smooth_down = np.zeros(image_arr.shape, dtype=bool)
    smooth_right[1:, :] = smooth_center[:-1, :]
    smooth_left[:-1, :] = smooth_center[1:, :]
    smooth_up[:, 1:] = smooth_center[:, :-1]
    smooth_down[:, :-1] = smooth_center[:, 1:]
    mask = (smooth_center | smooth_right | smooth_left | smooth_up | smooth_down)

    # Apply mask
    smooth_image_arr[mask] = 0.

    return smooth_image_arr


class Depthmap:
    """Depthmap and optional RGB

    Args:
        intrinsic (np.array): Camera intrinsic
        width (int): Width of the depthmap
        height (int): Height of the depthmap
        data (bytes): pixel_data
        depth_scale (float): Scalar to scale depthmap pixel to meters
        max_confidence (float): Confidence is amount of IR light reflected
                                (e.g. 0 to 255 in Lenovo, new standard is 0 to 7)
                                This is actually an int.
        device_pose (List[float]): The device pose (= position and rotation)
                              The ZIP-file header contains this pose
                              - `device_pose` is a list representation of this pose
                              - can be used to project into a different space
        rgb_fpath (str): Path to RGB file (e.g. to the jpg)
        rgb_array (np.array): RGB data
    """

    def __init__(
            self,
            intrinsics: np.ndarray,
            width: int,
            height: int,
            data: Optional[bytes],
            depthmap_arr: Optional[np.array],
            depth_scale: float,
            max_confidence: float,
            device_pose: List[float],
            rgb_fpath: Path,
            rgb_array: np.ndarray):
        """Constructor

        Either `data` or `depthmap_arr` has to be defined
        """
        self.width = width
        self.height = height

        self._intrinsics = intrinsics
        self.intrinsics = np.array(intrinsics)
        self.sensor = 1

        self.fx = self.intrinsics[self.sensor, 0] * self.width
        self.fy = self.intrinsics[self.sensor, 1] * self.height
        self.cx = self.intrinsics[self.sensor, 2] * self.width
        self.cy = self.intrinsics[self.sensor, 3] * self.height

        self.depth_scale = depth_scale
        self.max_confidence = max_confidence
        self.device_pose = device_pose
        self.device_pose_arr = np.array(device_pose).reshape(4, 4).T
        self.rgb_fpath = rgb_fpath
        self.rgb_array = rgb_array  # shape (height, width, 3)
        assert depthmap_arr is None or data is None
        self.depthmap_arr = self._parse_depth_data(data) if data else depthmap_arr

        # smoothing is only for normals, otherwise there is noise
        self.depthmap_arr_smooth = smoothen_depthmap_array(self.depthmap_arr)

        self.confidence_arr = self._parse_confidence_data(data) if data else None

    @property
    def has_rgb(self) -> bool:
        """Bool that indicates if the object has RGB data"""
        return self.rgb_array is not None

    @classmethod
    def create_from_zip(cls,
                        depthmap_dir: str,
                        depthmap_fname: str,
                        rgb_fname: str,
                        calibration_file: str) -> 'Depthmap':

        # read depthmap data
        path = extract_depthmap(depthmap_dir, depthmap_fname)
        with open(path, 'rb') as f:
            line = f.readline().decode().strip()
            header = line.split('_')
            res = header[0].split('x')
            width = int(res[0])
            height = int(res[1])
            depth_scale = float(header[1])
            max_confidence = float(header[2])
            if len(header) >= 10:
                position = (float(header[7]), float(header[8]), float(header[9]))
                rotation = (float(header[3]), float(header[4]), float(header[5]), float(header[6]))
                device_pose = matrix_calculate(position, rotation)
            else:
                device_pose = IDENTITY_MATRIX_4D
            data = f.read()
            f.close()

        # read rgb data
        if rgb_fname:
            rgb_fpath = Path(depthmap_dir) / 'rgb' / rgb_fname
            pil_im = Image.open(rgb_fpath)
            pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
            rgb_array = np.asarray(pil_im)
        else:
            rgb_fpath = rgb_fname
            rgb_array = None

        # read calibration file
        intrinsics = parse_calibration(calibration_file)
        depthmap_arr = None

        return cls(intrinsics,
                   width,
                   height,
                   data,
                   depthmap_arr,
                   depth_scale,
                   max_confidence,
                   device_pose,
                   rgb_fpath,
                   rgb_array
                   )

    @classmethod
    def create_from_array(cls,
                          depthmap_arr: np.ndarray,
                          rgb_arr: np.ndarray,
                          calibration_file: str) -> 'Depthmap':
        intrinsics = parse_calibration(calibration_file)
        height, width = depthmap_arr.shape
        data = None  # bytes
        depth_scale = 0.001
        max_confidence = 7.0
        device_pose = None
        rgb_fpath = None
        rgb_array = rgb_arr

        return cls(intrinsics,
                   width,
                   height,
                   data,
                   depthmap_arr,
                   depth_scale,
                   max_confidence,
                   device_pose,
                   rgb_fpath,
                   rgb_array,
                   )

    def calculate_normalmap_array(self, points_3d_arr: np.ndarray) -> np.ndarray:
        """Calculate normalmap consisting of normal vectors.

        A normal vector is based on a surface.
        The surface is constructed by a 3D point and it's neighbors.

        points_3d_arr: shape (3, width, height)

        Returns:
            3D points: shape (3, width, height)
        """

        # Get depth of the neighbor pixels
        dim_w = self.width - 1
        dim_h = self.height - 1
        depth_center = points_3d_arr[:, 1:, 1:].reshape(3, dim_w * dim_h)
        depth_x_minus = points_3d_arr[:, 0:-1, 1:].reshape(3, dim_w * dim_h)
        depth_y_minus = points_3d_arr[:, 1:, 0:-1].reshape(3, dim_w * dim_h)

        # Calculate a normal of the triangle
        vector_u = depth_center - depth_x_minus
        vector_v = depth_center - depth_y_minus

        normal = np.cross(vector_u, vector_v, axisa=0, axisb=0, axisc=0)

        normal = normalize(normal)

        normal = normal.reshape(3, dim_w, dim_h)

        # add black border to keep the dimensionality
        output = np.zeros((3, self.width, self.height))
        output[:, 1:, 1:] = normal
        return output

    def convert_2d_to_3d(self, sensor: int, x: float, y: float, depth: float) -> np.ndarray:
        """Convert point in pixels into point in meters

        Args:
            sensor: Index of sensor
            x
            y
            depth

        Returns:
            3D point
        """
        fx = self.intrinsics[sensor, 0] * self.width
        fy = self.intrinsics[sensor, 1] * self.height
        cx = self.intrinsics[sensor, 2] * self.width
        cy = self.intrinsics[sensor, 3] * self.height
        tx = (x - cx) * depth / fx
        ty = (y - cy) * depth / fy
        return np.array([tx, ty, depth])

    def convert_2d_to_3d_oriented(self, should_smooth: bool = False) -> np.ndarray:
        """Convert points in pixels into points in meters (and applying rotation)

        Args:
            should_smooth: Flag indicating weather to use a smoothed or an un-smoothed depthmap

        Returns:
            array of 3D points: shape(3, width, height)
        """
        depth = self.depthmap_arr_smooth if should_smooth else self.depthmap_arr  # shape: (width, height)

        xbig = np.expand_dims(np.array(range(self.width)), -1).repeat(self.height, axis=1)  # shape: (width, height)
        ybig = np.expand_dims(np.array(range(self.height)), 0).repeat(self.width, axis=0)  # shape: (width, height)

        # Convert point in pixels into point in meters
        tx = depth * (xbig - self.cx) / self.fx
        ty = depth * (ybig - self.cy) / self.fy
        dim4 = np.ones((self.width, self.height))
        res = np.stack([-tx, ty, depth, dim4], axis=0)

        # Transformation of point by device pose matrix
        points_4d = res.reshape((4, self.width * self.height))
        output = np.matmul(self.device_pose_arr, points_4d)
        output[0:2, :] = output[0:2, :] / abs(output[3, :])
        output = output.reshape((4, self.width, self.height))
        res = output[0:-1]

        # Invert y axis
        res[1, :, :] = -res[1, :, :]
        return res

    def segment_child(self, floor: float) -> np.ndarray:
        mask, segments = self.detect_objects(floor)

        # Select the most focused segment
        closest = sys.maxsize
        focus = -1
        for segment in segments:
            a = segment.aabb[0] - int(self.width / 2)
            b = segment.aabb[1] - int(self.height / 2)
            c = segment.aabb[2] - int(self.width / 2)
            d = segment.aabb[3] - int(self.height / 2)
            distance = a * a + b * b + c * c + d * d
            if closest > distance:
                closest = distance
                focus = segment.id

        mask = np.where(mask == focus, MASK_CHILD, mask)

        return mask

    def detect_floor(self, floor: float) -> np.ndarray:
        mask = np.zeros((self.width, self.height))
        assert self.depthmap_arr_smooth.shape == (self.width, self.height)
        mask[self.depthmap_arr_smooth == 0] = MASK_INVALID

        points_3d_arr = self.convert_2d_to_3d_oriented(should_smooth=True)
        normal = self.calculate_normalmap_array(points_3d_arr)

        cond1 = np.abs(normal[1, :, :]) > 0.5
        cond2 = (points_3d_arr[1, :, :] - floor) < 0.1
        per_pixel_cond = cond1 & cond2
        mask[per_pixel_cond] = MASK_FLOOR
        return mask

    def detect_objects(self, floor: float) -> Tuple[np.array, List[Segment]]:
        """Detect objects/children using seed algorithm

        Can likely not be used without for-loops over x,y

        Args:
            floor: Value of y-coordinate where the floor is

        Returns:
            mask (np.array): binary mask
            List[Segment]: a list of segments
        """
        current_id = -1
        segments = []
        dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        mask = self.detect_floor(floor)
        for x in range(self.width):
            for y in range(self.height):
                if mask[x, y] != 0:
                    continue
                pixel = [x, y]
                aabb = [pixel[0], pixel[1], pixel[0], pixel[1]]
                stack = [pixel]
                while len(stack) > 0:

                    # Get a next pixel from the stack
                    pixel = stack.pop()
                    depth_center = self.depthmap_arr[pixel[0], pixel[1]]

                    # Add neighbor points (if there is no floor and they are connected)
                    if mask[pixel[0], pixel[1]] == 0:
                        for direction in dirs:
                            pixel_dir = [pixel[0] + direction[0], pixel[1] + direction[1]]
                            depth_dir = self.depthmap_arr[pixel_dir[0], pixel_dir[1]]
                            if depth_dir > 0 and abs(depth_dir - depth_center) < 0.1:
                                stack.append(pixel_dir)

                    # Update AABB
                    aabb[0] = min(pixel[0], aabb[0])
                    aabb[1] = min(pixel[1], aabb[1])
                    aabb[2] = max(pixel[0], aabb[2])
                    aabb[3] = max(pixel[1], aabb[3])

                    # Update the mask
                    mask[pixel[0], pixel[1]] = current_id

                # Check if the object size is valid
                object_size_pixels = max(aabb[2] - aabb[0], aabb[3] - aabb[1])
                if object_size_pixels > self.width / 4:
                    segments.append(Segment(current_id, aabb))
                current_id = current_id - 1

        return mask, segments

    def get_angle_between_camera_and_floor(self) -> float:
        """Calculate an angle between camera and floor based on device pose"""
        centerx = int(self.width / 2)
        centery = int(self.height / 2)
        points_3d_arr = self.convert_2d_to_3d_oriented()  # shape: (3, width, height)
        # TODO revert to original code (depth=1 is important)

        point = points_3d_arr[:, centerx, centery]  # shape: (3,)
        angle = 90 + math.degrees(math.atan2(point[0], point[1]))
        return angle

    def get_floor_level(self) -> float:
        """Calculate an altitude of the floor in the world coordinates"""

        # Get normal vectors
        mask = np.zeros((self.width, self.height))
        assert self.depthmap_arr_smooth.shape == (self.width, self.height)
        mask[self.depthmap_arr_smooth == 0] = MASK_INVALID
        points_3d_arr = self.convert_2d_to_3d_oriented(should_smooth=True)
        normal = self.calculate_normalmap_array(points_3d_arr)

        cond = np.abs(normal[1, :, :]) > 0.5
        selection_of_points = points_3d_arr[1, :, :][cond]
        median = np.median(selection_of_points)
        return median

    def get_highest_point(self, mask: np.ndarray) -> np.ndarray:
        points_3d_arr = self.convert_2d_to_3d_oriented()
        y_array = np.copy(points_3d_arr[1, :, :])
        y_array[mask != MASK_CHILD] = -np.inf
        idx_highest_child_point = np.unravel_index(np.argmax(y_array, axis=None), y_array.shape)
        highest_point = points_3d_arr[:, idx_highest_child_point[0], idx_highest_child_point[1]]
        return highest_point

    def _parse_confidence_data(self, data) -> np.ndarray:
        """Parse depthmap confidence

        Returns:
            2D array of floats
        """
        output = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                output[x, y] = self._parse_confidence(data, x, y)
        return output

    def _parse_confidence(self, data: bytes, tx: int, ty) -> float:
        """Get confidence of the point in scale 0-1"""
        return data[(int(ty) * self.width + int(tx)) * 3 + 2] / self.max_confidence

    def _parse_depth_data(self, data) -> np.ndarray:
        output = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                output[x, y] = self._parse_depth(data, x, y)
        return output

    def _parse_depth(self, data: bytes, tx: int, ty: int) -> float:
        """Get depth of the point in meters"""
        if tx < 1 or ty < 1 or tx >= self.width or ty >= self.height:
            return 0.
        depth = data[(int(ty) * self.width + int(tx)) * 3 + 0] << 8
        depth += data[(int(ty) * self.width + int(tx)) * 3 + 1]
        depth *= self.depth_scale
        return depth


def convert_3d_to_2d(intrinsics: list, x: float, y: float, depth: float, width: int, height: int) -> list:
    """Convert point in meters into point in pixels

        Args:
            intrinsics of sensor: Tells if this is ToF sensor or RGB sensor
            x: X-pos in m
            y: Y-pos in m
            depth: distance from sensor to object at (x, y)
            width
            height

        Returns:
            tx, ty, depth
        """
    fx = intrinsics[0] * float(width)
    fy = intrinsics[1] * float(height)
    cx = intrinsics[2] * float(width)
    cy = intrinsics[3] * float(height)
    tx = x * fx / depth + cx
    ty = y * fy / depth + cy
    return [tx, ty, depth]


def parse_calibration(filepath: str) -> List[List[float]]:
    """Parse calibration file
    filepath: The content of a calibration file looks like this:
        Color camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
        Depth camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
    """
    with open(filepath, 'r') as f:
        calibration = []
        for _ in range(2):
            f.readline().strip()
            line_with_numbers = f.readline()
            intrinsic = parse_numbers(line_with_numbers)
            calibration.append(intrinsic)
    return calibration


def is_google_tango_resolution(width, height):
    """Check for special case for Google Tango devices with different rotation"""
    return width == 180 and height == 135


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Ensure the normal has a length of one

    This way of normalizing is commonly used for normals.
    It achieves that normals are of size 1.

    Args:
        vectors (np.array): Multiple vectors (e.g. could be normals)

    Returns:
        This achieves: abs(x) + abs(y) + abs(z) = 1
    """
    length = abs(vectors[0]) + abs(vectors[1]) + abs(vectors[2])
    return vectors / length
