import logging
from pathlib import Path
from typing import Union

import numpy as np

from depthmap import Depthmap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


def export_obj(fpath: Union[str, Path],
               dmap: Depthmap,
               floor_altitude_in_meters: float,
               triangulate: bool):
    """Export .obj file, which can be visualized in tools like Meshlab.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero
    triangulate=True generates OBJ of type mesh
    triangulate=False generates OBJ of type pointcloud
    """
    fpath = Path(fpath)
    count = 0
    indices = np.zeros((dmap.width, dmap.height))

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material_fpath = fpath.with_suffix('.mtl')
    if dmap.has_rgb:
        with open(material_fpath, 'w') as f:
            f.write('newmtl default\n')
            f.write(f'map_Kd {str(dmap.rgb_fpath.absolute())}\n')

    with open(fpath, 'w') as f:
        if dmap.has_rgb:
            f.write(f'mtllib {material_fpath.name}\n')
            f.write('usemtl default\n')

        points_3d_arr = dmap.convert_2d_to_3d_oriented()
        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.depthmap_arr[x, y]
                if not depth:
                    continue
                count = count + 1
                indices[x, y] = count  # add index of written vertex into array

                x_coord = points_3d_arr[0, x, y]
                y_coord = points_3d_arr[1, x, y] - floor_altitude_in_meters
                z_coord = points_3d_arr[2, x, y]
                f.write(f'v {x_coord} {y_coord} {z_coord}\n')
                f.write(f'vt {x / dmap.width} {y / dmap.height}\n')

        if triangulate:
            _do_triangulation(dmap, indices, f)
        logger.info('Mesh exported into %s', fpath)


def export_renderable_obj(fpath: Union[str, Path],
                          dmap: Depthmap,
                          floor_altitude_in_meters: float,
                          point_size_in_meters: float):
    """Export pointcloud as .obj file, which can be rendered in tools like Blender.

    floor_altitude_in_meters is the floor altitude to align floor to Y=zero
    point_size_in_meters is point size in meters
    """
    fpath = Path(fpath)
    count = 1

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material_fpath = fpath.with_suffix('.mtl')
    if dmap.has_rgb:
        with open(material_fpath, 'w') as f:
            f.write('newmtl default\n')
            f.write(f'map_Kd {str(dmap.rgb_fpath.absolute())}\n')

    with open(fpath, 'w') as f:
        if dmap.has_rgb:
            f.write(f'mtllib {material_fpath.name}\n')
            f.write('usemtl default\n')

        points_3d_arr = dmap.convert_2d_to_3d_oriented()
        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.depthmap_arr[x, y]
                if not depth:
                    continue

                x_coord = points_3d_arr[0, x, y]
                y_coord = points_3d_arr[1, x, y] - floor_altitude_in_meters
                z_coord = points_3d_arr[2, x, y]
                _write_obj_cube(f, dmap, x, y, count, x_coord, y_coord, z_coord, point_size_in_meters)
                count = count + 8

        logger.info('Mesh exported into %s', fpath)


def _do_triangulation(dmap: Depthmap, indices, filehandle):
    max_diff = 0.2
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            # get depth of all points of 2 potential triangles
            d00 = dmap.depthmap_arr[x, y]
            d10 = dmap.depthmap_arr[x + 1, y]
            d01 = dmap.depthmap_arr[x, y + 1]
            d11 = dmap.depthmap_arr[x + 1, y + 1]

            # check if first triangle points have existing indices
            if indices[x, y] > 0 and indices[x + 1, y] > 0 and indices[x, y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d00 - d10) + abs(d00 - d01) + abs(d10 - d01) < max_diff:
                    c = str(int(indices[x, y]))
                    b = str(int(indices[x + 1, y]))
                    a = str(int(indices[x, y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    _write_obj_triangle_indices(filehandle, a, b, c)

            # check if second triangle points have existing indices
            if indices[x + 1, y + 1] > 0 and indices[x + 1, y] > 0 and indices[x, y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d11 - d10) + abs(d11 - d01) + abs(d10 - d01) < max_diff:
                    a = str(int(indices[x + 1, y + 1]))
                    b = str(int(indices[x + 1, y]))
                    c = str(int(indices[x, y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    _write_obj_triangle_indices(filehandle, a, b, c)


def _write_obj_cube(
        f,
        dmap: Depthmap,
        x: int,
        y: int,
        count: int,
        x_coord: float,
        y_coord: float,
        z_coord: float,
        size: float):

    # cube points
    f.write(f'v {x_coord - size} {y_coord - size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord - size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord + size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord + size} {z_coord - size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord - size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord - size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord - size} {y_coord + size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')
    f.write(f'v {x_coord + size} {y_coord + size} {z_coord + size}\n')
    f.write(f'vt {x / dmap.width} {y / dmap.height}\n')

    # front face
    _write_obj_triangle_indices(f, str(count + 2), str(count + 1), str(count + 0))
    _write_obj_triangle_indices(f, str(count + 3), str(count + 2), str(count + 1))

    # back face
    _write_obj_triangle_indices(f, str(count + 4), str(count + 5), str(count + 6))
    _write_obj_triangle_indices(f, str(count + 7), str(count + 6), str(count + 5))

    # left face
    _write_obj_triangle_indices(f, str(count + 2), str(count + 4), str(count + 0))
    _write_obj_triangle_indices(f, str(count + 2), str(count + 4), str(count + 6))

    # right face
    _write_obj_triangle_indices(f, str(count + 1), str(count + 5), str(count + 3))
    _write_obj_triangle_indices(f, str(count + 7), str(count + 5), str(count + 3))

    # top face
    _write_obj_triangle_indices(f, str(count + 6), str(count + 3), str(count + 2))
    _write_obj_triangle_indices(f, str(count + 3), str(count + 6), str(count + 7))

    # bottom face
    _write_obj_triangle_indices(f, str(count + 0), str(count + 1), str(count + 4))
    _write_obj_triangle_indices(f, str(count + 4), str(count + 3), str(count + 1))


def _write_obj_triangle_indices(f, a: str, b: str, c: str):
    f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')


def _write_pcd_header(filehandle, count):
    filehandle.write('# timestamp 1 1 float 0\n')
    filehandle.write('# .PCD v.7 - Point Cloud Data file format\n')
    filehandle.write('VERSION .7\n')
    filehandle.write('FIELDS x y z c\n')
    filehandle.write('SIZE 4 4 4 4\n')
    filehandle.write('TYPE F F F F\n')
    filehandle.write('COUNT 1 1 1 1\n')
    filehandle.write('WIDTH ' + count + '\n')
    filehandle.write('HEIGHT 1\n')
    filehandle.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    filehandle.write('POINTS ' + count + '\n')
    filehandle.write('DATA ascii\n')


def export_pcd(filename: str, dmap: Depthmap):
    with open(filename, 'w') as f:
        count = str(_get_count(dmap))
        _write_pcd_header(f, count)

        for x in range(2, dmap.width - 2):
            for y in range(2, dmap.height - 2):
                depth = dmap.depthmap_arr[x, y]
                if not depth:
                    continue
                res = dmap.convert_2d_to_3d(1, x, y, depth)
                if not res.any():
                    continue
                confidence = dmap.confidence_arr[x, y]
                f.write(str(-res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + ' ' + str(confidence) + '\n')
        logger.info('Pointcloud exported into %s', filename)


def _get_count(dmap: Depthmap) -> int:
    count = 0
    for x in range(2, dmap.width - 2):
        for y in range(2, dmap.height - 2):
            depth = dmap.depthmap_arr[x, y]
            if not depth:
                continue
            res = dmap.convert_2d_to_3d(1, x, y, depth)
            if not res.any():
                continue
            count = count + 1
    return count
