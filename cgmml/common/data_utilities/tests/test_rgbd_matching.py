import pandas as pd
from pathlib import Path

from cgmml.common.data_utilities.rgbd_matching import (
    Artifact, FusedArtifact, match_depth_and_image_artifacts, match_df_with_depth_and_image_artifacts)


_COLUMN_NAMES = ['file_path', 'timestamp', 'scan_id', 'scan_step', 'height', 'weight', 'muac', 'order_number', 'format']


def test_match_depth_and_image_artifacts_no_match():

    depth_artifact_1 = Artifact(zip(_COLUMN_NAMES, [
        '1628138713366/f61c7b97-0aa0-4743-9c02-6b9f9ff7e97e',
        '2021-08-05 04:45:13.943297',
        '12737740-f5a8-11eb-882f-eb17b3ec776f',
        102,
        90,
        30,
        10,
        1,
        'depth']))
    assert len(_COLUMN_NAMES) == len(depth_artifact_1)

    depth_artifact_2 = Artifact(zip(_COLUMN_NAMES, [
        '1628138714772/ca02787a-ace0-4175-9a69-76a080763a8e',
        '2021-08-05 04:45:14.771368',
        '12737740-f5a8-11eb-882f-eb17b3ec776f',
        102,
        90,
        30,
        10,
        3,
        'depth']))

    image_artifact_1 = Artifact(zip(_COLUMN_NAMES, [
        '1628138771894/399fa5c4-1037-46d7-807d-b9360316aa84',
        '2021-08-05 04:46:11.943220',
        '130779e0-f5a8-11eb-882f-d391dbc35650',
        101,
        90,
        30,
        10,
        19,
        'rgb']))

    image_artifact_2 = Artifact(zip(_COLUMN_NAMES, [
        '1628138772442/ccdf04e7-7341-443c-960b-ab1b58749868',
        '2021-08-05 04:46:12.458855',
        '130779e0-f5a8-11eb-882f-d391dbc35650',
        101,
        90,
        30,
        10,
        20,
        'rgb']))

    image_artifacts = [image_artifact_1, image_artifact_2]
    depth_artifacts = [depth_artifact_1, depth_artifact_2]
    fused_artifacts = match_depth_and_image_artifacts(image_artifacts, depth_artifacts)
    assert len(fused_artifacts) == 0


def test_match_depth_and_image_artifacts_match():
    depth_artifact_1 = Artifact(file_path='depth1.txt', order_number=19)
    depth_artifact_2 = Artifact(file_path='depth2.txt', order_number=20)
    image_artifact_1 = Artifact(file_path='image1.txt', order_number=19)
    image_artifact_2 = Artifact(file_path='image2.txt', order_number=21)
    image_artifacts = [image_artifact_1, image_artifact_2]
    depth_artifacts = [depth_artifact_1, depth_artifact_2]
    fused_artifacts = match_depth_and_image_artifacts(image_artifacts, depth_artifacts)
    assert len(fused_artifacts) == 1, len(fused_artifacts)
    assert FusedArtifact(depth_artifact_1, image_artifact_1) in fused_artifacts


def test_fused_artifact2dict():
    df = pd.read_csv(Path(__file__).parent / "test_data/artifacts_image_and_depth.csv")
    fused_artifacts_dicts = match_df_with_depth_and_image_artifacts(df)
    assert len(fused_artifacts_dicts) == 44
    assert 'file_path_rgb' in fused_artifacts_dicts[0]
    assert 'file_path' in fused_artifacts_dicts[0]


if __name__ == "__main__":
    test_fused_artifact2dict()
