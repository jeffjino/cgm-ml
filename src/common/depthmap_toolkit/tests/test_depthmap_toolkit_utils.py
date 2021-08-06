import numpy as np

from common.depthmap_toolkit.depthmap_utils import matrix_transform_point


def test_matrix_transform_point_identity():
    device_pose = np.identity(4)
    device_pose_arr = np.array(list(device_pose.flatten())).reshape(4, 4).T
    point = [-0.006, -0.004, 1.0]
    output = matrix_transform_point(np.array(point), device_pose_arr)
    np.testing.assert_array_equal(output, np.array(point))


def test_matrix_transform_point():
    device_pose = np.identity(4) * -1.
    device_pose_arr = np.array(list(device_pose.flatten())).reshape(4, 4).T
    point = [-0.006, -0.004, 1.0]
    output = matrix_transform_point(np.array(point), device_pose_arr)
    np.testing.assert_array_equal(output, np.array([0.006, 0.004, -1.0]))


def test_matrix_transform_point_flipped():
    device_pose = np.identity(4) * -1.
    device_pose[:3, :3] = np.flip(device_pose[:3, :3], axis=1)
    device_pose_arr = np.array(list(device_pose.flatten())).reshape(4, 4).T
    point = [-0.006, -0.004, 1.0]
    output = matrix_transform_point(np.array(point), device_pose_arr)
    np.testing.assert_array_equal(output, np.array([-1.0, 0.004, 0.006]))


def test_matrix_transform_point_nonsym():
    device_pose = np.identity(4)
    device_pose[0, 2] = 1.
    device_pose_arr = np.array(list(device_pose.flatten())).reshape(4, 4).T
    point = [-0.5, 0.5, 1.0]
    output = matrix_transform_point(np.array(point), device_pose_arr)
    np.testing.assert_array_equal(output, np.array([-0.5, 0.5, 0.5]))


if __name__ == "__main__":
    test_matrix_transform_point_nonsym()
