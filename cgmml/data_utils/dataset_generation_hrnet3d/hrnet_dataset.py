import os
import sys
from tqdm import tqdm

from cgmml.common.evaluation.CV.csv_utils import read_csv
from cgmml.models.HRNET.body_pose import BodyPose

METADATA_ARTIFACT_ID = 2
METADATA_DEPTHMAP = 3
METADATA_RGB = 4
METADATA_SCAN_VERSION = 8
METADATA_SCAN_TYPE = 9


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('You did not enter raw data path, metadata file name')
        print('E.g.: python hrnet_dataset.py rawdata_dir metadata_file')
        sys.exit(1)

    path = sys.argv[1]
    calibration_file = '../../common/depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]
    output_path = 'dataset/'

    try:
        os.mkdir(output_path)
    except OSError:
        print('The directory already exists')

    metadata = read_csv(metadata_file)
    metadata_filtered = []
    for row in metadata:

        # Check if the scan version is good enough
        if row[METADATA_SCAN_VERSION].startswith('v0.8'):
            continue

        metadata_filtered.append(row)

    size = len(metadata_filtered)
    for index in tqdm(range(size)):
        row = metadata_filtered[index]

        # Get data paths
        depthmap_file = (path + row[METADATA_DEPTHMAP]).replace('"', '')
        rgb_file = (path + row[METADATA_RGB]).replace('"', '')
        output_file = f'{output_path}/{row[METADATA_ARTIFACT_ID]}.obj'.replace('"', '')

        # Process data
        try:
            body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
            if (body.get_person_count() != 1):
                continue
            body.export_object(output_file)
        except Exception:
            print(f'Data for {depthmap_file} does not seem to be valid')
