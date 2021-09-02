import logging
import random
import shutil
from importlib import import_module
from pathlib import Path

import tensorflow as tf
from azureml.core.run import Run

from constants import REPO_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


def copy_dir(src: Path, tgt: Path, glob_pattern: str, should_touch_init: bool = False):
    logger.info("Creating temp folder")
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True, exist_ok=True)
    if should_touch_init:
        (tgt / '__init__.py').touch(exist_ok=False)

    paths_to_copy = list(src.glob(glob_pattern))
    logger.info(f"Copying to {tgt} the following files: {str(paths_to_copy)}")
    for p in paths_to_copy:
        destpath = tgt / p.relative_to(src)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, destpath)


def is_offline_run(run: Run) -> bool:
    return run.id.startswith("OfflineRun")


# Get the current run.
RUN = Run.get_context()

from cgmml.common.evaluation.eval_utilities import (  # noqa: E402, F401
    is_ensemble_evaluation, is_multiartifact_evaluation)
from cgmml.common.evaluation.evaluation_classes import (  # noqa: E402, F401
    Evaluation, EnsembleEvaluation, MultiartifactEvaluation)
from cgmml.common.model_utils.run_initialization import OfflineRunInitializer, OnlineRunInitializer  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


QA_CONFIG_MODULES = [
    'qa_config_height',  # takes 14min in CI
#     'qa_config_height_clean',  # takes 9min in CI
    # 'qa_config_height_deep_ensemble',  # takes 50min in CI # is not in new central-india AzureML
    # 'qa_config_height_mcnn',  # takes 19min in CI   # is not in new central-india AzureML
    # 'qa_config_height_filter',  # is not in new central-india AzureML
#     'qa_config_weight',  # takes 13min in CI
]

if __name__ == "__main__":

    for qa_config_module in QA_CONFIG_MODULES:
        qa_config = import_module(qa_config_module)
        logger.info('Using the following config: %s', qa_config_module)

        model_config = qa_config.MODEL_CONFIG
        eval_config = qa_config.EVAL_CONFIG
        data_config = qa_config.DATA_CONFIG
        result_config = qa_config.RESULT_CONFIG
        filter_config = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None

        # Make experiment reproducible
        tf.random.set_seed(eval_config.SPLIT_SEED)
        random.seed(eval_config.SPLIT_SEED)

        if is_offline_run(RUN):
            output_csv_path = str(REPO_DIR / 'data' / result_config.SAVE_PATH)
            initializer = OfflineRunInitializer(data_config, eval_config)
        else:
            output_csv_path = result_config.SAVE_PATH
            initializer = OnlineRunInitializer(data_config, eval_config, RUN)

        if is_ensemble_evaluation(model_config):
            model_base_dir = (REPO_DIR / 'data' / model_config.EXPERIMENT_NAME) if is_offline_run(RUN) else Path('.')
            eval_class = EnsembleEvaluation
            descriptor = model_config.EXPERIMENT_NAME
        else:
            model_base_dir = REPO_DIR / 'data' / model_config.RUN_ID if is_offline_run(RUN) else Path('.')
            eval_class = MultiartifactEvaluation if is_multiartifact_evaluation(data_config) else Evaluation
            descriptor = model_config.RUN_ID
        evaluation = eval_class(model_config, data_config, model_base_dir, initializer.dataset_path)
        evaluation.get_the_model_path(initializer.workspace)

        # Get the QR-code paths
        qrcode_paths = evaluation.get_the_qr_code_path()
        if getattr(eval_config, 'DEBUG_RUN', False) and len(qrcode_paths) > eval_config.DEBUG_NUMBER_OF_SCAN:
            qrcode_paths = qrcode_paths[:eval_config.DEBUG_NUMBER_OF_SCAN]
            logger.info("Executing on %d qrcodes for FAST RUN", eval_config.DEBUG_NUMBER_OF_SCAN)

        dataset_evaluation, paths_belonging_to_predictions = evaluation.prepare_dataset(qrcode_paths, filter_config)
        prediction_array = evaluation.get_prediction_(evaluation.model_path_or_paths, dataset_evaluation)
        logger.info("Prediction made by model on the depthmaps...")
        logger.info(prediction_array)

        df = evaluation.prepare_dataframe(paths_belonging_to_predictions, prediction_array, result_config)
        evaluation.evaluate(df, result_config, eval_config, output_csv_path, descriptor)

        # Done.
        initializer.run.complete()
