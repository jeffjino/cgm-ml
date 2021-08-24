import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys

from csv_utils import read_csv
from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.visualisation import render_plot

METADATA_DEPTHMAP = 3

REPO_DIR = Path(__file__).parents[4]
EXPORT_DIR = REPO_DIR / 'data' / 'render'

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('You did not enter raw data path and metadata file path')
        print('E.g.: python evaluation.py rawdata_dir metadata_path')
        sys.exit(1)

    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    path = sys.argv[1]
    metadata_file = sys.argv[2]

    # Load metadata
    indata = read_csv(metadata_file)
    size = len(indata)

    # Re-create export folder
    try:
        shutil.rmtree(EXPORT_DIR)
    except BaseException:
        print('no previous data to delete')
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # For every depthmap show visualisation
    output = []
    processed = {}
    for index in range(size):
        data = indata[index]

        # Load data
        depthmap_file = path + data[METADATA_DEPTHMAP]
        depthmap_file = depthmap_file.replace('"', '')
        dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)

        # Render data
        file = str(EXPORT_DIR) + '/' + str(index + 1) + '.png'
        plt.imsave(file, render_plot(dmap))
