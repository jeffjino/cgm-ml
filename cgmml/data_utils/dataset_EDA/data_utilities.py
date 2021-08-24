import pandas as pd
import logging
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)


def convert_age_from_days_to_years(age_in_days: pd.Series) -> int:
    """Convert age in days into age in years"""
    age_in_years = age_in_days['age'] / 365
    return math.floor(age_in_years)


def extractqrcode(row: pd.Series) -> str:
    """Extract just the qrcode from them path"""
    return row['storage_path'].split('/')[1]


def draw_age_distribution(scans: pd.DataFrame):
    value_counts = scans['Years'].value_counts().sort_index(ascending=True)
    age_ax = value_counts.plot(kind='bar')
    age_ax.set_xlabel('age')
    age_ax.set_ylabel('no. of scans')
    logger.info(value_counts)


def draw_sex_distribution(scans: pd.DataFrame):
    value_counts = scans['sex'].value_counts().sort_index(ascending=True)
    ax = value_counts.plot(kind='bar')
    ax.set_xlabel('gender')
    ax.set_ylabel('no. of scans')
    logger.info(value_counts)


def _count_rows_per_age_bucket(artifacts):
    age_buckets = list(range(5))
    count_per_age_group = [artifacts[artifacts['Years'] == age].shape[0] for age in age_buckets]
    df_out = pd.DataFrame(count_per_age_group)
    df_out = df_out.T
    df_out.columns = age_buckets
    return df_out


def calculate_code_age_distribution(artifacts: pd.DataFrame, scan_type_colname: str) -> pd.DataFrame:
    logger.info(scan_type_colname)
    codes = list(artifacts[scan_type_colname].unique())
    dfs = []
    for code in codes:
        df = _count_rows_per_age_bucket(artifacts[artifacts[scan_type_colname] == code])
        df.rename(index={0: code}, inplace=True)
        dfs.append(df)
    result = pd.concat(dfs)
    result.index.name = 'codes'
    return result


def find_outliers(df: pd.DataFrame, column: str, condition: str, data_id_name: str) -> list:
    combined_condition = '@df.' + column + condition
    logger.info('Running the following query: %s', combined_condition)
    outlier_artifacts = df.query(combined_condition)
    outliers = []
    if data_id_name == 'qr':
        unique_outliers = outlier_artifacts.drop_duplicates(subset='qrcode', keep='first')
        outliers = unique_outliers.qrcode.tolist()
    elif data_id_name == 'id':
        unique_outliers = outlier_artifacts.drop_duplicates(subset='id', keep='first')
        outliers = unique_outliers.id.tolist()
    else:
        logger.info('data_id_name not valid')
    logger.info('No. of outliers: %d', len(outliers))
    return outliers


def display_images(images: list, grid_size: int, axes):
    current_file_number = 0
    for image_filename in images:
        x_position = current_file_number % grid_size
        y_position = current_file_number // grid_size
        plt_image = plt.imread(image_filename)
        rot_image = np.rot90(plt_image, 3)
        axes[x_position, y_position].imshow(rot_image)
        current_file_number += 1
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)


def display_depthmaps(depth: list, grid_size: int, axes):
    current_file_number = 0
    for depth_filename in depth:
        x_position = current_file_number % grid_size
        y_position = current_file_number // grid_size
        infile = open(depth_filename, 'rb')
        depthmap = pickle.load(infile)
        axes[x_position, y_position].imshow(np.squeeze(depthmap[0]))
        current_file_number += 1
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
