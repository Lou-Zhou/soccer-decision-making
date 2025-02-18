#trains success model from soccermap
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import mlflow
from xgboost import XGBClassifier, XGBRanker
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass.components import pass_success
from unxpass.components.utils import log_model, load_model
from unxpass.visualization import plot_action
STORES_FP = Path("../stores")
db_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
db = SQLiteDatabase(db_path)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/feature_testing"
dataset_train = partial(PassesDataset, path=custom_path)

pass_success_model = pass_success.SoccerMapComponent(model = pass_success.PytorchSoccerMapModel())
pass_success_model.train(dataset_train, trainer = {"accelerator": "cpu", "devices":1, "max_epochs": 50}, model_name = "val")
#can test with 
#dataset_train = partial(PassesDataset, DATA_DIR / "datasets" / "euro2020" / "test")
#pass_success_model.test(dataset_test)

mlflow.set_experiment("pass_success/soccermap")
with mlflow.start_run() as run:
    # Log the model
    mlflow.pytorch.log_model(pass_success_model.model, "model")

    # Retrieve the run ID
    run_id = run.info.run_id
    fail = run_id
    print(f"Pass Success Model saved with run_id: {run_id}")
outputstr = f"Pass Success Intended Model saved with run_id: {run_id}"
with open("successmodel_id.txt", "w") as text_file:
    text_file.write(outputstr)

"""
Can then load with:
model_pass_selection = pass_success.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'run_id', map_location='cpu'
    )
)"""


"""
xgboost model, originally used:

model = pass_success.XGBoostComponent(
    model=XGBClassifier(
        objective="binary:logistic", 
        eval_metric="auc"
    ),
    features={
        'startpolar': [
            'start_dist_to_goal_a0',
            'start_angle_to_goal_a0'
        ],
        'relative_startlocation': [
            'start_dist_goalline_a0',
            'start_dist_sideline_a0'
        ],
        'endpolar': [
            'end_dist_to_goal_a0',
            'end_angle_to_goal_a0'
        ],
        'relative_endlocation': [
            'end_dist_goalline_a0',
            'end_dist_sideline_a0'
        ],
        'movement': [
            'movement_a0',
            'dx_a0',
            'dy_a0'
        ],
        'angle': [
            'angle_a0'
        ],
        'ball_height_onehot': [
            'ball_height_ground_a0',
            'ball_height_low_a0',
            'ball_height_high_a0'
        ],
        'under_pressure': [
            'under_pressure_a0'
        ],
        'dist_defender': [
            'dist_defender_start_a0',
            'dist_defender_end_a0',
            'dist_defender_action_a0'
        ],
        'nb_opp_in_path': [
            'nb_opp_in_path_a0'
        ]
    }, 
)
model.train(dataset_train)

mlflow.set_experiment(experiment_name="pass_success/xgb")
modelinfo = log_model(model, artifact_path="component")
print(f"Model saved as {modelinfo.model_uri}")
#can then load using
#model = load_model(modelinfo.model_uri) 
"""