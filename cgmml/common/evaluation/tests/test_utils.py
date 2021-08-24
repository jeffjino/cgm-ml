import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from evaluation.eval_utilities import (COLUMN_NAME_GOODBAD,  # noqa: E402
                                       calculate_percentage_confusion_matrix,
                                       draw_uncertainty_goodbad_plot)
from model_utils.utils import get_dataset_path  # noqa: E402


def test_draw_uncertainty_goodbad_plot():
    uncertainties = [1.010987, 1.073083, 1.312352, 3.515901, 1.602865]
    goodbad = [1.0, 0.0, 1.0, 0.0, 0.0]

    df = pd.DataFrame(list(zip(uncertainties, goodbad)), columns=['uncertainties', COLUMN_NAME_GOODBAD])
    with tempfile.NamedTemporaryFile() as temp:
        draw_uncertainty_goodbad_plot(df, temp.name)


def test_calculate_percentage_confusion_matrix():
    data = np.array([[2, 4, 0],
                     [2, 2, 1],
                     [1, 1, 2]])
    trues, fp, fn = calculate_percentage_confusion_matrix(data)
    assert trues == 40
    assert fp == 33.33
    assert fn == 26.67


def test_get_dataset_path():
    result = get_dataset_path(Path("/tmp/data/"), "test")
    assert result == '/tmp/data/test'
