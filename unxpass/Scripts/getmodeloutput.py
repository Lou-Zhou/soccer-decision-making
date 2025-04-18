from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from tqdm import tqdm
import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds, pass_value_speeds_testing
from unxpass.visualization import plot_action
from unxpass.ratings_custom import LocationPredictions

from matplotlib.backends.backend_pdf import PdfPages

from unxpass.converters import playVisualizers
#wspeed ce9da14c0e88476aaab3cac209b7659d
model_pass_wSpeeds = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/4e97c61c9b6749719c2bdf3d05292a07/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive = True
)

model_pass_noSpeed = pass_value_custom.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/1b4acefc54cd40208f77239061443d00/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive = True
)

#59e582d607844c31bb20b58578b90688
#No C,D: 20e7d3695d7049d0a513922d32b44a11

path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
db = SQLiteDatabase(path)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Hawkeye_Features/Hawkeye_Features_Updated"
dataset_test = partial(PassesDataset, path=custom_path)
sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
surfaces_wSpeed= model_pass_wSpeeds.predict_surface(dataset_test, db = None, model_name = "val")
surfaces_nSpeed = model_pass_noSpeed.predict_surface(dataset_test, db = None, model_name = "val")
pdf_filename = "value_test.pdf"#value model is straight up predicting 0... :(
features_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Hawkeye_Features/Hawkeye_Features_Updated"
freeze_frame = pd.read_parquet(f"{features_dir}/x_freeze_frame_360.parquet")
speed = pd.read_parquet(f"{features_dir}/x_speed.parquet")
start = pd.read_parquet(f"{features_dir}/x_startlocation.parquet")
with PdfPages(pdf_filename) as pdf:
    for idx, row in tqdm(sequences.iterrows()):
        id = row['id']
        game_id = row['match_id']
        action_id = row['index']
        ff_action = freeze_frame.loc[(game_id, action_id)]
        start_action = start.loc[(game_id, action_id)]
        speed_action = speed.loc[(game_id, action_id)]
        title = f"Game {game_id}, statsbomb id {id}, index {action_id}"
        fig = playVisualizers.plot_single_model_comparison_features(game_id, action_id, ff_action, start_action, speed_action, 
            title, surfaces_wSpeed, surfaces_nSpeed, subtitle_1 = "With Speed", subtitle_2 = "Without Speed", xg = False, log = False)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

