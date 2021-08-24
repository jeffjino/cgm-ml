import sys
from csv_utils import read_csv, write_csv

METADATA_SCAN_ID = 0
METADATA_ORDER = 1
METADATA_ARTIFACT_ID = 2
METADATA_FILEPATH = 3
METADATA_EXTENSION = 4


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('You did not enter metadata file path')
        print('E.g.: python rgbd_match.py metadata_path')
        sys.exit(1)

    metadata_file = sys.argv[1]

    # Create a map
    indata = read_csv(metadata_file)
    size = len(indata)
    mapping = {}
    for index in range(1, size):
        data = indata[index]
        if data[METADATA_EXTENSION] != '.jpg':
            continue
        scanid = data[METADATA_SCAN_ID]
        order = data[METADATA_ORDER]
        key = scanid + str(order)
        mapping[key] = data

    # For every depthmap add rgb file
    output = []
    processed = {}
    for index in range(1, size):
        data = indata[index]
        if data[METADATA_EXTENSION] != '.depth':
            continue
        scanid = data[METADATA_SCAN_ID]
        order = data[METADATA_ORDER]
        key = scanid + str(order)
        if processed.get(key):
            continue

        # add the RGB-D match
        if mapping.get(key):
            data[METADATA_EXTENSION] = mapping[key][METADATA_FILEPATH]
            print(str(index) + '/' + str(size) + ' - ' + scanid + ',' + str(order))
            output.append(data)
            processed[key] = True
    write_csv('newmetadata.csv', output)
