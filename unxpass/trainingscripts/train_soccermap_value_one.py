from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom
from unxpass.components.utils import log_model, load_model
import warnings
import time
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset, SamePassesDataset
from unxpass.components import pass_value, pass_value_custom, pass_success, pass_success_custom
pd.options.mode.chained_assignment = None

DATA_DIR = Path("../stores/")
dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
feat_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_fail"
dataset_test = partial(PassesDataset, path = feat_path)
val_model = pass_value_custom.SoccerMapComponent(model = pass_value_custom.PytorchSoccerMapModel(), offensive = False)
print(val_model.initialize_dataset(dataset_test, model_name = "val").labels)
#time.sleep(10)
print("training...")
val_model.train(dataset_test, model_name = 'val', trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(val_model.model, "model")

        #Retrieve the run ID
    run_id = run.info.run_id
    fail_def = run_id
    print(f"Diagnostic Value/Success Model saved with run_id: {run_id}")
outputstr = f"Diagnostic Backwards Value/Success model saved with run_id: {run_id}"
with open("valmodel_special_id.txt", "w") as text_file:
    text_file.write(outputstr)

