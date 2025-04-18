#train soccermap selection
"""
Originally ran from command line as: 
unxpass train \
  $(pwd)/config \
  $(pwd)/stores/datasets/euro2020/train\
  experiment="pass_selection/soccermap"
"""
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset, SamePassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.components.withSpeeds import pass_selection_speeds
pd.options.mode.chained_assignment = None

DATA_DIR = Path("../stores/")
STORES_FP = Path("../stores")
dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
db = SQLiteDatabase(dbpath)
#/home/lz80/un-xPass/stores/datasets/custom/dataset_subset5
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers"
#dataset_train = partial(PassesDataset, path=custom_path)
#dataset_test = partial(PassesDataset, path=custom_path)

dataset_train = partial(PassesDataset, path=custom_path)
dataset_test = partial(PassesDataset, path=custom_path)


pass_selection_model = pass_selection_speeds.SoccerMapComponent(model = pass_selection_speeds.PytorchSoccerMapModel())


pass_selection_model.train(dataset_train, model_name = "sel", trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})


mlflow.set_experiment("pass_selection/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(pass_selection_model.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    fail = run_id
    print(f"Selection Model saved with run_id: {run_id}")
outputstr = f"Selection Model saved with run_id: {run_id}"
with open("selmodel_id.txt", "w") as text_file:
    text_file.write(outputstr)
#selection model: 23c126e3f89d41f5b5bacafbed36c66f
"""
Can then load with:
model_pass_selection = pass_selection.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'run_id', map_location='cpu'
    )
)"""
#I will probably need to adjust end and start locations