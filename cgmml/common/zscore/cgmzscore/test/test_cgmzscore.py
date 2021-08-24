import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.main import z_score_lhfa, z_score_wfa, z_score_wfh, z_score_wfl, calculate_sam_mam  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


df = pd.read_csv(str(Path(__file__).parents[0]) + '/testdata.csv')

# 1 for male
# 2 for female


def check_null(i, df):
    return (pd.isnull(df.loc[i, 'WEIGHT'])
            or pd.isnull(df.loc[i, '_agedays'])
            or pd.isnull(df.loc[i, '_ZWEI'])
            or df.loc[i, '_agedays'] == 0
            or pd.isnull(df.loc[i, '_ZLEN'])
            or pd.isnull(df.loc[i, '_ZWFL']))


def test_z_score_wfa():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.9f}". format(df['WEIGHT'][i]))
        v = z_score_wfa(weight=str(g), sex=sex, age_in_days=str((df['_agedays'][i])))
        ans = float("{0:.2f}". format(abs(v - df['_ZWEI'][i])))
        assert ans <= 0.01


def test_z_score_lhfa():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.9f}". format(df['HEIGHT'][i]))

        v = z_score_lhfa(height=str(g), sex=sex, age_in_days=str((df['_agedays'][i])))

        logger.info(g)

        ans = float("{0:.2f}". format(abs(v - df['_ZLEN'][i])))
        assert ans <= 0.01


def test_z_score_wfh():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = z_score_wfh(height=str(g), weight=str(t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v - df['_ZWFL'][i])))
        assert ans <= 0.01


def test_z_score_wfl():
    for i in range(len(df)):
        if check_null(i, df):
            continue

        sex = 'M' if df['GENDER'][i] == 1 else 'F'

        g = float("{0:.5f}". format(df['HEIGHT'][i]))
        t = float("{0:.9f}". format(df['WEIGHT'][i]))

        v = z_score_wfl(height=str(g), weight=str(t), sex=sex, age_in_days=str((df['_agedays'][i])))

        ans = float("{0:.2f}". format(abs(v - df['_ZWFL'][i])))
        assert ans <= 0.01


def test_calculate_sam_mam():
    diagnosis = calculate_sam_mam(weight="10.4", muac="14.89", age_in_days="683", sex='F', height="84.80")
    assert diagnosis == 'Healthy'
