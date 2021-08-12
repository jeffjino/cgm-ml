from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

from cgmml.common.data_utilities.mlpipeline_utils import create_layers, ArtifactProcessor

DATA_UTILITIES_DIR = Path(__file__).parents[1].absolute()
ARTIFACT_ZIP_PATH = 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a/depth/bd67cd9e-69c7-11eb-984b-77ac9d2b4986'
ARTIFACT_TUPLE = (
    ARTIFACT_ZIP_PATH,
    '2021-04-22_13-34-33-302557',
    'c571de02-a723-11eb-8845-bb6589a1fbe8',
    102,
    50.0,
    20.555,
    10.0,
    3,
)
_COLUMN_NAMES = ['file_path', 'timestamp', 'scan_id', 'scan_step', 'height', 'weight', 'muac', 'order_number']
IDX2COL = {i: col for i, col in enumerate(_COLUMN_NAMES)}


def test_artifact_processor():
    input_dir = DATA_UTILITIES_DIR / 'tests' / 'zip_files'

    with TemporaryDirectory() as output_dir:
        artifact_processor = ArtifactProcessor(input_dir, output_dir, IDX2COL)
        processed_fname = artifact_processor.process_artifact_tuple(ARTIFACT_TUPLE)

        depthmap, targets = pickle.load(open(processed_fname, 'rb'))
        assert depthmap.shape == (240, 180, 1), depthmap.shape
        assert 'height' in targets

        pickle_path_expected = str(
            DATA_UTILITIES_DIR
            / 'tests'
            / 'pickle_files'
            / 'scans'
            / 'c571de02-a723-11eb-8845-bb6589a1fbe8'
            / '102'
            / 'pc_c571de02-a723-11eb-8845-bb6589a1fbe8_2021-04-22_13-34-33-302557_102_3.p')
        assert pickle_path_expected.split('/')[-4:] == processed_fname.split('/')[-4:]


def test_create_layers():
    zip_input_full_path = DATA_UTILITIES_DIR / 'tests' / 'zip_files' / ARTIFACT_ZIP_PATH
    layers, metadata = create_layers(zip_input_full_path)
    assert layers.shape == (240, 180, 1), layers.shape

    assert isinstance(metadata['raw_header'], str), metadata['raw_header']
    expected_header = '320x240_0.001_7_-0.15561539_-0.07175923_-0.6638096_0.72800505_-8.440511_0.3684988_-1.3508477'
    assert metadata['raw_header'] == expected_header
