import pandas as pd
from data_utilities import find_outliers


QR_CODE_1 = "1585013006-yqwb95138e"
QR_CODE_2 = "1555555555-yqqqqqqqqq"
QR_CODE_3 = "1212121212-jajajajaja"
QR_CODE_4 = "9922992299-lelelelele"


def prepare_test_df():
    data = {
        'artifacts': [
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_2}/100/pc_{QR_CODE_2}_1591849321035_100_000.p',
            f'scans/{QR_CODE_3}/100/pc_{QR_CODE_3}_1591849321035_100_000.p',
            f'scans/{QR_CODE_1}/100/pc_{QR_CODE_1}_1591849321035_100_000.p',
            f'scans/{QR_CODE_4}/100/pc_{QR_CODE_4}_1591849321035_100_000.p'],
        'age': [180, 365, 3000, 180, 20],
        'weight': [3.0, 6.0, 32.0, 3.0, 2.7],
        'height': [30.0, 95.0, 102.0, 30.0, 195.0],
        'qrcode': [QR_CODE_1, QR_CODE_2, QR_CODE_3, QR_CODE_1, QR_CODE_4],
    }
    df = pd.DataFrame.from_dict(data)
    return df


def test_find_outlier_qrcodes_age_min():
    df = prepare_test_df()
    qrs = find_outliers(df, 'age', '<365/2', data_id_name='qr')
    assert (len(qrs) == 2)


def test_find_outlier_qrcodes_age_max():
    df = prepare_test_df()
    qrs = find_outliers(df, 'age', '>365*6', data_id_name='qr')
    assert (len(qrs) == 1)


def test_find_outlier_qrcodes_weight_min():
    df = prepare_test_df()
    qrs = find_outliers(df, 'weight', '<5.0', data_id_name='qr')
    assert (len(qrs) == 2)


def test_find_outlier_qrcodes_weight_max():
    df = prepare_test_df()
    qrs = find_outliers(df, 'weight', '>30.0', data_id_name='qr')
    assert (len(qrs) == 1)


def test_find_outlier_qrcodes_height_min():
    df = prepare_test_df()
    qrs = find_outliers(df, 'height', '<40.0', data_id_name='qr')
    assert (len(qrs) == 1)


def test_find_outlier_qrcodes_height_max():
    df = prepare_test_df()
    qrs = find_outliers(df, 'height', '>150.0', data_id_name='qr')
    assert (len(qrs) == 1)
