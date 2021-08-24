#!/usr/bin/env bash

set -euox pipefail

python convertdepth2pcd.py honor/ honor/camera_calibration.txt
python convertpcd2depth.py export honor/camera_calibration.txt # --res 240x180
python toolkit.py output honor/camera_calibration.txt


# python toolkit.py testdata/lenovo_phab2pro_180x135/ testdata/lenovo_phab2pro_180x135/camera_calibration.txt


# python toolkit.py tests/huawei_p40pro tests/huawei_p40pro/camera_calibration.txt
# python toolkit.py tests/huawei_p40pro_full tests/huawei_p40pro_full/camera_calibration.txt
