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
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.components.utils import load_model
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds#, pass_value_speeds_testing
from unxpass.visualizers import plotPlays
from matplotlib.backends.backend_pdf import PdfPages
import configparser
import sys

# Handle file paths ----

config = configparser.ConfigParser()
config.read('soccer-decision-making.ini')
path_data = config['path']['data']

# Handle command line arguments ----

run_id = sys.argv[1]
path_output = sys.argv[2]


#wspeed ce9da14c0e88476aaab3cac209b7659d
model_pass_wSpeeds = pass_selection_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        "runs:/" + run_id + "/model", map_location = "cpu"
    )
)#2ddca7aa73c946e0a162bf5843b66c66
#model_checkpoint_path = "/home/lz80/un-xPass/unxpass/Scripts/trainingscripts/lightning_logs/version_94/checkpoints/epoch=24-step=864825.ckpt"
#loaded_model = pass_value_speeds.PytorchSoccerMapModel.load_from_checkpoint(model_checkpoint_path)
#model_pass_wSpeeds = pass_value_speeds.SoccerMapComponent(model = loaded_model)
#59e582d607844c31bb20b58578b90688
#No C,D: 20e7d3695d7049d0a513922d32b44a11

#features_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_filtered"
features_dir = path_data + "/Hawkeye/Hawkeye_Features/sequences"
dataset_test = partial(PassesDataset, path=features_dir)
#sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv", delimiter = ";")
#game_id = "DFL-MAT-J03YJD"
surfaces_wSpeed= model_pass_wSpeeds.predict_surface(dataset_test, db = None)
#surfaces_wSpeed = model_pass_wSpeeds.predict_surface(dataset_test, db = None)

freeze_frame = pd.read_parquet(f"{features_dir}/x_freeze_frame_360.parquet")
speed = pd.read_parquet(f"{features_dir}/x_speed.parquet")
end = pd.read_parquet(f"{features_dir}/x_endlocation.parquet")
start = pd.read_parquet(f"{features_dir}/x_startlocation.parquet")
surface_type = {"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}
count = 0
with PdfPages(path_output) as pdf:
    for game_id in surfaces_wSpeed:
        for action_id in tqdm(surfaces_wSpeed[game_id]):
            ff_action = freeze_frame.loc[(game_id, action_id)]
            start_action = start.loc[(game_id, action_id)]
            speed_action = speed.loc[(game_id, action_id)]
            end_action = end.loc[(game_id, action_id)]
            surface = surfaces_wSpeed[game_id][action_id]
            title = f"Game {game_id}, index {action_id}, max_val {np.amax(surface)}"
            # 
            fig = plotPlays.visualize_surface(ff_action, start_action, end_action, title = title, surface = surface, surface_kwargs= surface_type, log = False, modelType = "sel")
            #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
            count = count + 1
            if count > 200:
                break

