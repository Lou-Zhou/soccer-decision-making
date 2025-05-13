# %%
"""
Getting all selection / evaluation criterions - legacy code
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

db = SQLiteDatabase("/home/lz80/un-xPass/stores/hawkeye_all.sql")
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features"
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

# %%
rater = LocationPredictions(
    pass_selection_component=model_pass_selection,
    pass_success_component=model_pass_success,
    pass_value_success_offensive_component=model_pass_value_success_offensive,
    pass_value_fail_offensive_component = model_pass_value_fail_offensive,
    pass_value_success_defensive_component=model_pass_value_success_defensive,
    pass_value_fail_defensive_component = model_pass_value_fail_defensive,
)
test = rater.rate_all_games(db, dataset_test, summarize = True)#this will take a bit, might want to make this more efficient in the future


# %%
test.to_csv("/home/lz80/rdf/sp161/shared/soccer-decision-making/eurotest_womens_criterias_reception.csv")


