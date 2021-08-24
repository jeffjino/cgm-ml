from collections import namedtuple
from copy import copy
import logging
from typing import List, Tuple

from bunch import Bunch
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

IMAGE_TARGET_WIDTH = 180
IMAGE_TARGET_HEIGHT = 240
ORDER_DIFFERENCE_ALLOWED = 0  # works for scan version v0.9+

FusedArtifact = namedtuple('FusedArtifact', 'depth_artifact image_artifact')


class Artifact(Bunch):
    pass


def find_closest_image(image_order_numbers, depth_order_number):
    """Find corresponding image for the given depthmap on the basis of order"""
    closest_image_order_number = min(image_order_numbers, key=lambda image_order: abs(image_order - depth_order_number))
    return closest_image_order_number


def find_image_artifact_with_order_number(image_artifacts: List[Artifact], order_number: str) -> Artifact:
    """Find the image artifact with the given order_number"""
    for artifact in image_artifacts:
        if artifact.order_number == order_number:
            return artifact


def check_fields_are_equal(image_artifact: Artifact, depth_artifact: Artifact) -> None:
    """Check scan_id, height, weight, muac, scan_step, order_number are equal"""
    fields_supposed_to_be_equal = list(depth_artifact.keys())
    fields_to_ignore = ['file_path', 'timestamp', 'format']
    for f in fields_to_ignore:
        if f in fields_supposed_to_be_equal:
            fields_supposed_to_be_equal.remove(f)

    for field in fields_supposed_to_be_equal:
        depth_artifact_value = getattr(depth_artifact, field)
        image_artifact_value = getattr(image_artifact, field)
        if depth_artifact_value != image_artifact_value:
            raise ValueError(f"{field} is not equal for depthmap and image artifact")


def match_depth_and_image_artifacts(image_artifacts: List[Artifact],
                                    depth_artifacts: List[Artifact]) -> List[FusedArtifact]:
    fused_artifacts = []
    image_order_numbers = [image_artifact.order_number for image_artifact in image_artifacts]

    for depth_artifact in depth_artifacts:
        closest_image_order_number = find_closest_image(image_order_numbers, depth_artifact.order_number)
        if abs(closest_image_order_number - depth_artifact.order_number) > ORDER_DIFFERENCE_ALLOWED:
            logger.debug(f"No corresponding image found for depthmap with order_number {depth_artifact.order_number}")
            continue
        image_artifact = find_image_artifact_with_order_number(image_artifacts, closest_image_order_number)
        check_fields_are_equal(image_artifact, depth_artifact)
        fused_artifacts.append(FusedArtifact(depth_artifact, image_artifact))
    return fused_artifacts


def fused_artifact2dict(fused_artifact: FusedArtifact) -> dict:
    out_fused_artifact = copy(fused_artifact.depth_artifact)
    image_artifact = fused_artifact.image_artifact

    out_fused_artifact['file_path_rgb'] = image_artifact.file_path
    return dict(out_fused_artifact)


def match_df_with_depth_and_image_artifacts(df: pd.DataFrame) -> List[dict]:
    """Process a dataframe with the depthmap and the corresponding image"""
    all_fused_artifacts = []
    scan_ids = df.scan_id.unique()

    for scan_id in scan_ids:
        df_scan = df[df.scan_id == scan_id]
        assert len(df_scan.scan_step.unique()) == 1

        df_depth = df_scan.loc[df.format == 'depth']
        df_image = df_scan.loc[df.format == 'rgb']

        depths: List[Tuple[str]] = list(df_depth.itertuples(index=False, name=None))
        images: List[Tuple[str]] = list(df_image.itertuples(index=False, name=None))

        depth_artifacts = [Artifact(zip(df.columns, d)) for d in depths]
        image_artifacts = [Artifact(zip(df.columns, i)) for i in images]

        fused_artifacts = match_depth_and_image_artifacts(image_artifacts, depth_artifacts)
        fused_artifacts_dicts = [fused_artifact2dict(a) for a in fused_artifacts]
        all_fused_artifacts.extend(fused_artifacts_dicts)
    return all_fused_artifacts
