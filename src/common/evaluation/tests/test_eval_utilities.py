from pathlib import Path
from bunch import Bunch

import pytest
import pandas as pd

from common.evaluation.eval_utilities import tf_load_pickle, prepare_sample_dataset

CWD = Path(__file__).resolve()
REPO_DIR = str(CWD.parents[4].absolute())
DATA_CONFIG = Bunch(dict(
    NAME='dataset-anontest',
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    BATCH_SIZE=16,
    NORMALIZATION_VALUE=7.5,
    TARGET_INDEXES=[0],
    CODES=['100', '101', '102', '200', '201', '202'],
))


def test_prepare_sample_dataset():
    dataset_path = str("testfiles")
    qrcode_list = [
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
    ]
    scantype_list = ["100", "100", "100", "102"]
    artifact_list = [
        "pc_1585004725-18cqo1np0j_1592801845251_100_000.p",
        "pc_1585004725-18cqo1np0j_1592801845251_100_001.p",
        "pc_1585012629-ac1ippx2qy_1591848606827_100_000.p",
        "pc_1585012629-ac1ippx2qy_1591848606827_102_072.p",
    ]
    prediction_list = [96.8, 96.8, 85.3, 84.8]
    target_list = [95.5, 95.5, 85.0, 85.0]
    columns = ['qrcode', 'artifact', 'scantype', 'GT', 'predicted']

    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': target_list,
        'predicted': prediction_list
    }, columns=columns)
    df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

    dataset_sample = prepare_sample_dataset(df_sample, dataset_path, DATA_CONFIG)
    assert len(dataset_sample) == 3, 'There should be 3 samples in the dataset'


def test_tf_load_pickle():
    pickle_path = str(REPO_DIR
                      + "/src/common/data_utilities/tests/pickle_files/scans/c571de02-a723-11eb-8845-bb6589a1fbe8/102/"
                      + "pc_c571de02-a723-11eb-8845-bb6589a1fbe8_2021-04-22_13-34-33-302557_102_3.p")

    test_image = tf_load_pickle(pickle_path, DATA_CONFIG.NORMALIZATION_VALUE, DATA_CONFIG)

    assert isinstance(test_image, tuple), 'The type of object should be a tuple'
    assert test_image[1].shape[0] == DATA_CONFIG.IMAGE_TARGET_HEIGHT
    assert test_image[1].shape[1] == DATA_CONFIG.IMAGE_TARGET_WIDTH


def test_tf_load_not_a_pickle():
    wrong_path = str(REPO_DIR + "/src/common/data_utilities/tests/zip_files/be1faf54-"
                     + "69c7-11eb-984b-a3ffd42e7b5a/depth/bd67cd9e-69c7-11eb-984b-77ac9d2b4986")
    normalization_value = 7.5
    with pytest.raises(Exception, match='UnpicklingError'):
        tf_load_pickle(wrong_path, normalization_value, DATA_CONFIG)


def test_tf_load_empty_pickle():
    empty_path = str(REPO_DIR + "/src/common/evaluation/QA/eval_depthmap_models/src/testfiles/empty_pickle_file.p")
    normalization_value = 7.5
    with pytest.raises(Exception):
        tf_load_pickle(empty_path, normalization_value, DATA_CONFIG)
