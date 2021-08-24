import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import argparse
import os
import logging
import glob2 as glob
from gapnet.models import GAPNet
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pickle
from preprocessing import preprocess_pointcloud, preprocess_targets
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

# Parse the arguments.
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument(
    "--run_id",
    type=str,
    required=True,
    help="The id of the run that contains the trained model.")
args = parser.parse_args()


# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logger.info('Running in offline mode...')

    # Access workspace.
    logger.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "gapnet-offline")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")

    # Get dataset.
    logger.info('Accessing dataset...')
    if os.path.exists("premiumfileshare") == False:
        dataset_name = "cgmmldevpremium-SampleDataset-Example"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='.', overwrite=False)
    dataset_path = glob.glob(os.path.join("premiumfileshare", "*"))[0]

# Online run. Use dataset provided by training notebook.
else:
    logger.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]


# Download the model from the provided run.
logger.info('Downloading model from run with id %d', args.run_id)


# Locate the run that contains the model.
run_that_contains_model = None
for experiment_run in experiment.get_runs():
    if experiment_run.id == args.run_id:
        run_that_contains_model = experiment_run
        break
if run_that_contains_model is None:
    logger.info('ERROR! Run not found!')
    exit(0)

# Download the model.
logger.info('Downloading the model...')
output_directory = "model-" + args.run_id
run_that_contains_model.download_files(output_directory=output_directory)

# Instantiate the model with its weights.
logger.info('Creating the model...')
model = GAPNet()
logger.info('Loading model weights...')
model.load_weights(os.path.join(output_directory, "gapnet_weights.h5"))
model.summary()

# Get all files from the dataset.
logger.info('Find all files for evaluation...')
pickle_files = glob.glob(os.path.join(dataset_path, "**", "*.p"))

# Evaluate all files.
# TODO parallelize this.
logger.info('Evaluate all files...')
data = {"results": []}
for index, pickle_file in enumerate(pickle_files):
    name = os.path.basename(pickle_file).split(".")[0]

    # Load and preprocess the data.
    pointcloud, targets = pickle.load(open(pickle_file, "rb"))
    pointcloud = np.array([preprocess_pointcloud(pointcloud, 1024, list(range(3)))])
    #targets = preprocess_targets(targets, [0])[0] # 0 is height
    targets = preprocess_targets(targets, [1])[0]  # 1 is weight
    predicted_targets = model.predict(pointcloud)[0][0]

    # Store results.
    result = {
        "name": name,
        "targets": str(targets),
        "predicted_targets": str(predicted_targets),
        "error": str(np.abs(predicted_targets - targets))
    }
    data["results"].append(result)
    if index % 1000 == 0:
        print("{} per cent".format(int(100 * (index / len(pickle_files)))))

# Save results.
logger.info('Saving results...')
json_path = "results.json"
with open(json_path, 'w') as outfile:
    json.dump(data, outfile)

# Upload to azure.
logger.info('Uploading results to Azure...')
run.upload_file(name=json_path, path_or_stream=json_path)

# Done.
logger.info('Done.')
run.complete()
