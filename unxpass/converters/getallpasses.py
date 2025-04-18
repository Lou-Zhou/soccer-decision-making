#Script for xgboost model which predicts odds for each teammate
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets_custom import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.ratings_all import AllPredictions
STORES_FP = Path("../stores")

db = SQLiteDatabase(STORES_FP / "database.sql")

dataset_test = partial(PassesDataset, path=STORES_FP / "datasets" / "euro2020" / "test")
plt_settings = {"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}

model_pass_selection = load_model("runs:/5a13feeb1f8b45078e40aaa944b17979/component")
#model_pass_selection.test(dataset_test)
model_pass_success = load_model('runs:/f977aaf2f5a0497cb51f5e730ae64609/component')

#model_pass_success.test(dataset_test)
model_pass_value_success = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/f94362f83b6c4f2caa5da826daaacb8d/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
model_pass_value_fail = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/f6f93a33c3ac4bc6a3b29442c0908dc7/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
#model_pass_success.test(dataset_test)
#model_pass_value.test(dataset_test)
options = pd.read_parquet("/home/lz80/un-xPass/stores/datasets/euro2020/x_pass_options.parquet")
true_picked = pd.read_parquet("/home/lz80/un-xPass/stores/datasets/euro2020/y_receiver.parquet")
rater = AllPredictions(
    pass_selection_component=model_pass_selection,
    pass_success_component=model_pass_success,
    pass_value_success_component=model_pass_value_success,
    pass_value_fail_component = model_pass_value_fail
)
all_dfs = []
all_sels = model_pass_selection.predict(dataset_test)
for action_id in options.index.get_level_values("action_id").unique()[1:100]:
    sels = all_sels.loc[(3795506, action_id)]
    true = true_picked.loc[(3795506, action_id)]
    sels_df = pd.DataFrame({"selection_probability":sels, "pass_options":list(sels.index)})
    true_df = pd.DataFrame({"receiver":true["receiver"], "pass_options":list(true.index)})
    action_pass = rater.rate(db, dataset_test, 3795506, action_id, options)
    action_pass = pd.merge(action_pass, sels_df)
    action_pass = pd.merge(action_pass, true_df)
    all_dfs.append(action_pass)
all_combined = pd.concat(all_dfs)
all_combined.to_csv("euro2020finalpasses.csv")