#train soccermap value
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom
from unxpass.components.utils import log_model, load_model
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_value, pass_value_custom
from unxpass.components.withSpeeds import pass_value_speeds, pass_value_speeds_testing
pd.options.mode.chained_assignment = None

DATA_DIR = Path("../stores/")
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers"
fail_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers_fail"
success_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers_success"

dataset_success = partial(PassesDataset, path = success_path)
dataset_fail = partial(PassesDataset, path = fail_path)
dataset_total = partial(PassesDataset, path = custom_path)#628 vs 2747
STORES_FP = Path("../stores")

value_fail_offensive = pass_value_speeds.SoccerMapComponent(model = pass_value_speeds.PytorchSoccerMapModel(), offensive = True)
print("training...")
value_fail_offensive.train(dataset_fail, trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10}, model_name = 'val')
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(value_fail_offensive.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    fail_off = run_id
    print(f"Fail Offensive Model saved with run_id: {run_id}")

#Defensive Fail
value_fail_defensive = pass_value_speeds.SoccerMapComponent(model = pass_value_speeds.PytorchSoccerMapModel(), offensive = False)

value_fail_defensive.train(dataset_fail, model_name = 'val', trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
#     Log the model
    mlflow.pytorch.log_model(value_fail_defensive.model, "model")

        #Retrieve the run ID
    run_id = run.info.run_id
    fail_def = run_id
    print(f"Fail Defensive Model saved with run_id: {run_id}")
# Offensive Success
value_success_offensive = pass_value_speeds.SoccerMapComponent(model = pass_value_speeds.PytorchSoccerMapModel(), offensive = True)

value_success_offensive.train(dataset_success,model_name = 'val', trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(value_success_offensive.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    success_off = run_id
    print(f"Success Offensive Model saved with run_id: {run_id}")
# Defensive Success
value_success_defensive = pass_value_speeds.SoccerMapComponent(model = pass_value_speeds.PytorchSoccerMapModel(), offensive = False)

value_success_defensive.train(dataset_success, model_name = 'val', trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 10})
mlflow.set_experiment("pass_value/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(value_success_defensive.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    success_def = run_id
    print(f"Success Defensive Model saved with run_id: {run_id}")




print(f"Fail Offensive Model saved as {fail_off}")
print(f"Fail Defensive Model saved as {fail_def}")
print(f"Success Offensive Model saved as {success_off}")
print(f"Success Defensive Model saved as {success_def}")
outputstr = f"Fail Offensive Model saved as {fail_off}\n" + f"Fail Defensive Model saved as {fail_def}\n" + f"Success Offensive Model saved as {success_off}\n" +f"Success Defensive Model saved as {success_def}"
with open("valmodel_id.txt", "w") as text_file:
    text_file.write(outputstr)
#success offensive: 5e7b6c613cd84f3e91702a73a36eebe6
#success defensive: fc10352d1c4948b4ac556443e4de575a
#fail offensive: 88a3c3e2ea6441c288d237c50ca542a0   
#fail defensive:  28946631a0b54fe89dc83ca129815b4f  #real!!


"""
Can then load with:
model_pass_selection = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'run_id', map_location='cpu'
    )
)"""

##Debugging 50:
#1 .use many epochs
#2. for every incomplete pass - calculate distance from incomplete pass to nearest opponent
#- if that distance is below some threshold, add .5 xg to response - for both success and fail
#3.for every potential recipient - closest defender, find player who is wide open
#4. parametric value surface used as bayesian prior 
# 4. Drawing board: focus on dataset that un-xpass dataset uses