# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Dataset using Databricks
# MAGIC 
# MAGIC Steps
# MAGIC * databricks driver gets list of artifacts from postgres DB
# MAGIC * databricks driver copies artifacts (from blob stoarage to DBFS)
# MAGIC * databricks workers process artifacts
# MAGIC * databricks driver uploads all the blobs (from DBFS to blob storage)
# MAGIC 
# MAGIC ![dataset creation workflow](https://dev.azure.com/cgmorg/e5b67bad-b36b-4475-bdd7-0cf6875414df/_apis/git/repositories/465970a9-a8a5-4223-81c1-2d3f3bd4ab26/Items?path=%2F.attachments%2Fdataset_creation_steps-7d790a55-684f-407a-a6db-9238789ff761.png&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1&sanitize=true&versionDescriptor.version=wikiMaster)

# COMMAND ----------

# flake8: noqa

! pip install scikit-image
! pip install azure-storage-blob

! pip install cgm-ml-common==3.1.6

# COMMAND ----------

from datetime import datetime, timezone
from multiprocessing.dummy import Pool as ThreadPool
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import psycopg2
from azure.storage.blob import BlobServiceClient

from cgmml.common.data_utilities.mlpipeline_utils import ArtifactProcessor
from cgmml.common.data_utilities.rgbd_matching import match_df_with_depth_and_image_artifacts

# COMMAND ----------

# Constants
ENV_PROD = "env_prod"
ENV_QA = "env_qa"
ENV_SANDBOX = "env_dev"

# COMMAND ----------

# Configuration
ENV = ENV_SANDBOX

DBFS_DIR = f"/tmp/{ENV}"

DATASET_TYPE = 'depthmap'  # Supported: 'rgbd' and 'depthmap'

DATA_CATEGORY = 'Train'  # Supported: 'Train' and 'Test'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Secret scopes
# MAGIC 
# MAGIC For reading secrets, we use [secret scopes](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes).
# MAGIC 
# MAGIC When we create scopes in each environment (sandbox, demo/qa, prod), we name it `cgm-ml-scope`.

# COMMAND ----------

SECRET_SCOPE = "cgm-ml-scope"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access SQL database to find all the scans/artifacts of interest
# MAGIC 
# MAGIC #### SQL query
# MAGIC 
# MAGIC We build our SQL query, so that we get all the required information for the ML dataset creation:
# MAGIC - the artifacts (depthmap, RGB, pointcloud)
# MAGIC - the targets (measured height, weight, and MUAC)
# MAGIC 
# MAGIC The ETL packet shows which tables are involved
# MAGIC 
# MAGIC ![image info](https://dev.azure.com/cgmorg/e5b67bad-b36b-4475-bdd7-0cf6875414df/_apis/git/repositories/465970a9-a8a5-4223-81c1-2d3f3bd4ab26/Items?path=%2F.attachments%2Fcgm-solution-architecture-etl-draft-ETL-samplling-71a42e64-72c4-4360-a741-1cfa24622dce.png&download=false&resolveLfs=true&%24format=octetStream&api-version=5.0-preview.1&sanitize=true&versionDescriptor.version=wikiMaster)
# MAGIC 
# MAGIC The query will produce one artifact per row.

# COMMAND ----------

host = dbutils.secrets.get(scope=SECRET_SCOPE, key="mlapi-db-host")
user = dbutils.secrets.get(scope=SECRET_SCOPE, key="mlapi-db-user")
password = dbutils.secrets.get(scope=SECRET_SCOPE, key="mlapi-db-pw")

conn = psycopg2.connect(host=host, database='cgm-ml', user=user, password=password)

# COMMAND ----------

sql_cursor = conn.cursor()

# COMMAND ----------

SQL_QUERY_BASE = f"""
    SELECT DISTINCT(f.file_path), f.created as timestamp,
           s.id as scan_id, s.scan_type_id as scan_step, s.version as scan_version,
           m.height, m.weight, m.muac,
           a.ord as order_number, a.format,
           di.model as device_model,
           p.id as person_id, p.age, p.sex
    FROM file f
    INNER JOIN artifact a ON f.id = a.file_id
    INNER JOIN scan s     ON s.id = a.scan_id
    INNER JOIN measure m  ON m.person_id = s.person_id AND m.head_circumference IS NOT Null
    INNER JOIN person p ON p.id = s.person_id
    INNER JOIN child_data_category cdc ON p.id = cdc.person_id
    INNER JOIN data_category dc ON dc.id = cdc.data_category_id
    INNER JOIN device_info di ON di.id = s.device_info_id
    WHERE dc.description = '{DATA_CATEGORY}'
    AND di.model = 'HUAWEI VOG-L29'"""

if DATASET_TYPE == 'depthmap':
    SQL_QUERY = f"""{SQL_QUERY_BASE}
    AND a.format IN ('depth', 'application/zip');"""
elif DATASET_TYPE == 'rgbd':
    SQL_QUERY = f"""{SQL_QUERY_BASE}
    AND a.ord IS NOT NULL;"""
else:
  raise NameError(f'Unknown dataset type: {DATASET_TYPE}')

sql_cursor.execute(SQL_QUERY)

# Get multiple query_result rows
NUM_ARTIFACTS = 300  # None
query_results_tmp: List[Tuple[str]] = sql_cursor.fetchall() if NUM_ARTIFACTS is None else sql_cursor.fetchmany(NUM_ARTIFACTS)

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation of a file_path**
# MAGIC 
# MAGIC The SQL result provides file_paths which have this format
# MAGIC 
# MAGIC ```
# MAGIC Example: '1618896404960/2fe0ee0e-daf0-45a4-931e-cfc7682e1ce6'
# MAGIC Format: f'{unix-timestamp}/{random uuid}'
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataframe

# COMMAND ----------

_column_names: Tuple[psycopg2.extensions.Column] = sql_cursor.description
df = pd.DataFrame(query_results_tmp, columns=list(map(lambda x: x.name, _column_names)))

def clean_timestamp(timestamp_str: str):
    return timestamp_str.replace(' ', '-').replace(':', '-'). replace('.', '-')
df['timestamp'] = df['timestamp'].astype(str).apply(clean_timestamp)

# COMMAND ----------

# hack to insert fake manual measurements in case there are none
if 'muac' not in df.columns:
    df.insert(4, 'muac', 10)
if 'weight' not in df.columns:
    df.insert(4, 'weight', 30)
if 'height' not in df.columns:
    df.insert(4, 'height', 90)

# COMMAND ----------

print("Shape with duplicates:", df.shape)
df = df.drop_duplicates(subset=['scan_id', 'scan_step', 'timestamp', 'order_number'])
print("Shape without duplicates:", df.shape)
df.head()

# COMMAND ----------

# Filter children between 0.5 years and 5 years
df = df.loc[(df['age']>=365/2) & (df['age']<=365*5)]
print("Shape within age range:", df.shape)
df.head()

# COMMAND ----------

print("Unique persons:", len(df.person_id.unique()))
print("Unique scan_ids:", len(df.scan_id.unique()))
print("Unique artifacts:", len(df.file_path.unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Download artifact files to DBFS
# MAGIC 
# MAGIC In order for databricks to process the blob data, we need to transfer it to the DBFS of the databricks cluster.
# MAGIC 
# MAGIC Note:
# MAGIC * Copying from mount is very very slow, therefore we copy the data
# MAGIC 
# MAGIC ## Download blobs
# MAGIC 
# MAGIC We use [Manage blobs Python SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#download-blobs)
# MAGIC to download blobs directly from the Storage Account(SA) to [DBFS](https://docs.databricks.com/data/databricks-file-system.html).

# COMMAND ----------

CONNECTION_STR_SRC_SA = dbutils.secrets.get(scope=SECRET_SCOPE, key="mlapi-sa-sas-token")
# STORAGE_ACCOUNT_NAME = dbutils.secrets.get(scope=SECRET_SCOPE, key="mlapi-sa-name")
CONTAINER_NAME_SRC_SA = "cgm-result"

# COMMAND ----------

BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STR_SRC_SA)

# COMMAND ----------

def download_from_blob_storage(src: str, dest: str, container: str):
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container, blob=src)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as download_file:
        content = blob_client.download_blob().readall()
        download_file.write(content)

# COMMAND ----------

# Gather file_paths
_file_paths = df['file_path'].tolist()

# Remove duplicates
_file_paths = list(set(_file_paths))
print(f"Number of files to download: {len(_file_paths)}")

# COMMAND ----------

# Download
def _download(full_name):
    download_from_blob_storage(src=full_name, dest=f"/dbfs{DBFS_DIR}/{full_name}", container=CONTAINER_NAME_SRC_SA)

# COMMAND ----------

NUM_THREADS = 8
pool = ThreadPool(NUM_THREADS)
results = pool.map(_download, _file_paths)

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform ZIP into pickle
# MAGIC 
# MAGIC Here we document the format of the artifact path
# MAGIC 
# MAGIC ```
# MAGIC f"scans/1583462505-43bak4gvfa/101/pc_1583462505-43bak4gvfa_1591122173510_101_002.p"
# MAGIC f"qrcode/{scan_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"
# MAGIC ```
# MAGIC 
# MAGIC Idea for a future format could be to include person_id like so:
# MAGIC ```
# MAGIC f"qrcode/{person_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Match depthmap and rgb to rgbd

# COMMAND ----------

if DATASET_TYPE == 'rgbd':
    fused_artifacts_dicts = match_df_with_depth_and_image_artifacts(df)
    df_to_process = pd.DataFrame(fused_artifacts_dicts)
else:
    df_to_process = df

# COMMAND ----------

df_to_process

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Dataframe to list of query_result_dicts
# MAGIC query_result_dicts are needed for processing

# COMMAND ----------

query_results_dicts: List[dict] = df_to_process.to_dict('records')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process

# COMMAND ----------

input_dir = f"/dbfs{DBFS_DIR}"
output_dir = f"/dbfs{DBFS_DIR}"
artifact_processor = ArtifactProcessor(input_dir, output_dir, dataset_type=DATASET_TYPE, should_rotate_rgb=False)

# COMMAND ----------

def map_fct(query_result_dict):
#     print("the_query_dict", query_result_dict)
    return query_result_dict, artifact_processor.create_and_save_pickle(query_result_dict)

# COMMAND ----------

# Processing all artifacts at once
rdd = spark.sparkContext.parallelize(query_results_dicts)
print(rdd.getNumPartitions())
rdd_processed = rdd.map(map_fct)
processed_dicts_and_fnames = rdd_processed.collect()
print(processed_dicts_and_fnames[:3])

# COMMAND ----------

_ = df_to_process['scan_step'].value_counts().plot(kind='pie')

# COMMAND ----------

df_to_process.shape

# COMMAND ----------

print(f"All processed samples: {len(processed_dicts_and_fnames)}")

# Select successfully processed
processed_dicts_with_success = []
processed_fnames_with_success = []
for query_result_dict, fn in processed_dicts_and_fnames:
    if fn != '':
        processed_dicts_with_success.append(query_result_dict)
        processed_fnames_with_success.append(fn)

# Update dataframe
df_to_process = pd.DataFrame(processed_dicts_with_success)
processed_fnames = processed_fnames_with_success

print(f"Successfully pickled samples: {len(processed_fnames_with_success)}")

# COMMAND ----------

# Check if processed_fnames has duplicates before uploading
assert len(set(processed_fnames)) == len(processed_fnames)

# COMMAND ----------

# MAGIC %md
# MAGIC # Upload to blob storage
# MAGIC 
# MAGIC We use [Manage blobs Python SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#upload-blobs-to-a-container)
# MAGIC to upload blobs

# COMMAND ----------

CONNECTION_STR_DEST_SA = dbutils.secrets.get(scope=SECRET_SCOPE, key="dset-sa-sas-token")
# STORAGE_ACCOUNT_NAME_DEST_SA = dbutils.secrets.get(scope=SECRET_SCOPE, key="dset-sa-name")
CONTAINER_NAME_DEST_SA = "cgm-datasets"
BLOB_SERVICE_CLIENT_DSET = BlobServiceClient.from_connection_string(CONNECTION_STR_DEST_SA)

# COMMAND ----------

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix): return text[len(prefix):]
    return text

PREFIX = f"/dbfs{DBFS_DIR}/"

def upload_to_blob_storage(src: str, dest_container: str, dest_fpath: str):
    blob_client = BLOB_SERVICE_CLIENT_DSET.get_blob_client(container=dest_container, blob=dest_fpath)
    with open(src, "rb") as data:
        blob_client.upload_blob(data, overwrite=False)

# COMMAND ----------

DATASET_NAME = "dataset"
dest_dir = datetime.now(timezone.utc).strftime(f"{DATASET_NAME}-{DATASET_TYPE}-{DATA_CATEGORY}-%Y-%m-%d-%H-%M-%S")

def _upload(full_name):
    assert PREFIX in full_name, full_name
    dest_fpath = os.path.join(dest_dir, remove_prefix(full_name, PREFIX))
    upload_to_blob_storage(src=full_name, dest_container=CONTAINER_NAME_DEST_SA, dest_fpath=dest_fpath)

NUM_THREADS = 8
pool = ThreadPool(NUM_THREADS)
results = pool.map(_upload, processed_fnames)

# COMMAND ----------

labels_filename = f"{output_dir}/labels.csv"
df_to_process.to_csv(labels_filename, index=False)

# COMMAND ----------

dest_fpath = os.path.join(dest_dir, remove_prefix(labels_filename, PREFIX))
upload_to_blob_storage(src=labels_filename, dest_container=CONTAINER_NAME_DEST_SA, dest_fpath=dest_fpath)

# COMMAND ----------


