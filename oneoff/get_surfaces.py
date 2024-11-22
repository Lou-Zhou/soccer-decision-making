# %%
"""
Getting all selection / evaluation criterions
"""

from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success,pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.ratings_custom import LocationPredictions
STORES_FP = Path("../stores")

db = SQLiteDatabase("/home/lz80/un-xPass/stores/hawkeye_whole.sql")
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2"
dataset_test = partial(PassesDataset, path=custom_path)
plt_settings = {"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}

model_pass_selection = pass_selection.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/04d45112c139473590b5049cb3797d0d/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
#model_pass_selection.test(dataset_test)
#model_pass_success = load_model('runs:/f977aaf2f5a0497cb51f5e730ae64609/component')
model_pass_success = pass_success.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/78b7ab86dc864858b8814fe811b8796a/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)

#
model_pass_value_success_offensive = pass_value_custom.SoccerMapComponent(#good
    model=mlflow.pytorch.load_model(
        'runs:/ec44b8ef79944b10a0d87d13949a1fd3/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )# 7 channel model
)

model_pass_value_success_defensive = pass_value_custom.SoccerMapComponent(#good
    model=mlflow.pytorch.load_model(
        'runs:/fc10352d1c4948b4ac556443e4de575a/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive = False#rest are 9 channel models
)

model_pass_value_fail_offensive = pass_value_custom.SoccerMapComponent(#ntbc
    model=mlflow.pytorch.load_model(
        'runs:/a7eb2344aa414ef582bbc06dcea6b00e/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)

model_pass_value_fail_defensive = pass_value_custom.SoccerMapComponent(#ntbc
    model=mlflow.pytorch.load_model(
        'runs:/28946631a0b54fe89dc83ca129815b4f/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive = False
)

#model_pass_success.test(dataset_test)
#model_pass_value.test(dataset_test)
#options = pd.read_parquet("/home/lz80/un-xPass/stores/datasets/euro2020/x_pass_options.parquet")
#true_picked = pd.read_parquet("/home/lz80/un-xPass/stores/datasets/euro2020/y_receiver.parquet")
def group_agg(input_df):
    samples = input_df
    samples["split"] = samples['original_event_id'].str.split('-')
    samples['home_pass'] = samples['original_event_id'].str.rsplit('-', 1).str[0]
    samples["idx"] = samples['original_event_id'].str.rsplit('-', 1).str[1]
    samples = samples[samples['idx'].str.len() < 3]
    result = samples.groupby('home_pass').agg({
        'game_id': 'first',           # average for this column
        'action_id': 'first',         # first value for this column
        'start_x': 'first',
        'start_y': 'first',
        'selection_criterion':'mean'
    }).assign(numFrames=samples.groupby('home_pass').size())
    return result
# %%
rater = LocationPredictions(
    pass_selection_component=model_pass_selection,
    pass_success_component=model_pass_success,
    pass_value_success_offensive_component=model_pass_value_success_offensive,
    pass_value_fail_offensive_component = model_pass_value_fail_offensive,
    pass_value_success_defensive_component=model_pass_value_success_defensive,
    pass_value_fail_defensive_component = model_pass_value_fail_defensive,
)
games = db.games().index
output_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEyeResults_2"
output_dir_verbose = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEyeResults_2_verbose"
for game_id in games:
    #d5d702c7-e650-46a7-a154-5109ccb6e8d0
    print(f"Game ID: {game_id}")
    test = rater.rate_one_game(db, dataset_test, game_id)
    test_agg = group_agg(test)
    print(f"Saving to {output_dir}/{game_id}_test.csv")
    test_agg.to_csv(f"{output_dir}/{game_id}.csv")
    test.to_csv(f"{output_dir_verbose}/{game_id}.csv")
