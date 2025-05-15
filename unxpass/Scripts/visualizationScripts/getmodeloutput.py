#Script to plot model outputs
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from tqdm import tqdm
import numpy as np
import mlflow
from scipy.ndimage import zoom
import torch
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds, pass_value_speeds_testing
from unxpass.visualization import plot_action
from unxpass.ratings_custom import LocationPredictions
from unxpass.visualizers import plotPlays
from matplotlib.backends.backend_pdf import PdfPages

from unxpass.converters import playVisualizers
#wspeed ce9da14c0e88476aaab3cac209b7659d
model_pass_wSpeeds = pass_success_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/3c974bbccb0e40b8a8eeab8e91ff9821/model', map_location = 'cpu'
    )
)#2ddca7aa73c946e0a162bf5843b66c66

#59e582d607844c31bb20b58578b90688
#No C,D: 20e7d3695d7049d0a513922d32b44a11

custom_path = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye_Features/Hawkeye_Features_Updated"
dataset_test = partial(PassesDataset, path=custom_path)
sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv", delimiter = ";")
surfaces_wSpeed= model_pass_wSpeeds.predict_surface(dataset_test, db = None)
#surfaces_nSpeed = model_pass_noSpeed.predict_surface(dataset_test, db = None, model_name = "sel")
pdf_filename = "success_test_player.pdf"
freeze_frame = pd.read_parquet(f"{features_dir}/x_freeze_frame_360.parquet")
speed = pd.read_parquet(f"{features_dir}/x_speed.parquet")
start = pd.read_parquet(f"{features_dir}/x_startlocation.parquet")
with PdfPages(pdf_filename) as pdf:
    for idx, row in tqdm(sequences.iloc[0:500].iterrows()):
        id = row['id']
        game_id = row['match_id']
        action_id = row['index']
        ff_action = freeze_frame.loc[(game_id, action_id)]
        start_action = start.loc[(game_id, action_id)]
        speed_action = speed.loc[(game_id, action_id)]
        title = f"Game {game_id}, statsbomb id {id}, index {action_id}"
        surface = surfaces_wSpeed[game_id][action_id]
        # 
        fig = plotPlays.visualize_coords_from_tracking_2(ff_action, start_action, speed_action, (game_id, action_id), title = title, surface = surface, surface_kwargs={"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, playerOnly = True, modelType = "val")
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

