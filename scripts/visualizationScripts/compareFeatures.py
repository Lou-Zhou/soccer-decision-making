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

#wspeed ce9da14c0e88476aaab3cac209b7659d
model_pass_wSpeeds = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/f73621e21a064dd4bcb199efce9b4d26/model', map_location = 'cpu'
    ), offensive = False, success = False
)#2ddca7aa73c946e0a162bf5843b66c66

#59e582d607844c31bb20b58578b90688
#No C,D: 20e7d3695d7049d0a513922d32b44a11

features_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/oldFeatures/all_features_defl_fail"
freeze_frame = pd.read_parquet(f"{features_dir}/x_freeze_frame_360.parquet")
start = pd.read_parquet(f"{features_dir}/x_startlocation.parquet")
endloc = pd.read_parquet(f"{features_dir}/x_endlocation.parquet")
concedes_xg = pd.read_parquet(f"{features_dir}/y_concedes_xg.parquet")
speed = pd.read_parquet(f"{features_dir}/x_speed.parquet")

new_feats = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_failed"
end_n = pd.read_parquet(f"{new_feats}/x_endlocation.parquet")
start_n = pd.read_parquet(f"{new_feats}/x_startlocation.parquet")
concedes_xg_n = pd.read_parquet(f"{new_feats}/y_concedes_xg.parquet")
freeze_frame_n = pd.read_parquet(f"{new_feats}/x_freeze_frame_360.parquet")
speed_n = pd.read_parquet(f"{new_feats}/x_speed.parquet")

def getLocs(action, end, start, concedes_xg, freeze_frame, speed):
    ff_action = freeze_frame.loc[action]
    start_action = start.loc[action]
    end_action = end.loc[action]
    speed_action = speed.loc[action]
    concedes_xg_action = concedes_xg.loc[action]
    return ff_action, start_action, end_action, concedes_xg_action, speed_action
db = SQLiteDatabase("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql")
#dataset_test = partial(PassesDataset, path=features_dir)
#surfaces_wSpeed= model_pass_wSpeeds.predict_surface(dataset_test, db = None)
#surfaces_nSpeed = model_pass_noSpeed.predict_surface(dataset_test, db = None, model_name = "sel")
pdf_filename = "feat_comparison.pdf"
with PdfPages(pdf_filename) as pdf:
    currentGame = None
    action_map = {}
    for game_id, action_id in tqdm(concedes_xg.index):
        idx_old = (game_id, action_id)
        if game_id != currentGame:
            actions = db.actions(game_id = game_id)
            currentGame = game_id
            action_map = {idx : actions.loc[idx]["original_event_id"] for idx in actions.index}
        playId = int(float(action_map[idx_old]))
        if playId not in end_n.index.get_level_values(1):
            continue
        idx_new = (game_id, playId)
        new_concedes = concedes_xg_n.loc[idx_new]['concedes_xg']
        old_concedes = concedes_xg.loc[idx_old]['concedes_xg']
        if new_concedes == old_concedes:
            continue
        ff_action, start_action, end_action, concedes_xg_action, speed_action = getLocs(idx_new, end_n, start_n, concedes_xg_n, freeze_frame_n, speed_n)
        ff_action_old, start_action_old, end_action_old, concedes_xg_action_old, speed_action_old = getLocs(idx_old, endloc, start, concedes_xg, freeze_frame, speed)
        title = f"Game {game_id}, playId {playId}, action_id {action_id}"
        subtitle_1 = f"Concedes xG: {new_concedes}"
        subtitle_2 = f"Concedes xG old: {old_concedes}"
        
        fig, axs = plt.subplots(1,2, figsize=(10,8))
        fig.suptitle(title, fontsize=9)
        plotPlays.visualize_play_from_parquet(ff_action, start_action, end_action, idx_new, speed_action, title = subtitle_1, ax = axs[0])
        plotPlays.visualize_play_from_parquet(ff_action_old, start_action_old, end_action_old, idx_old, speed_action_old, title = subtitle_2, ax = axs[1])
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

