#get animations for all events in the index_dict.json file - usually used to animate weird events as found in visualizeFeatures.py
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from scipy.ndimage import zoom
import warnings
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.components.withSpeeds import pass_value_speeds
from unxpass.components.utils import load_model
from matplotlib.backends.backend_pdf import PdfPages
import mplsoccer
from unxpass.visualizers import plotPlays, Animations
from tqdm import tqdm
import json
import ast
import traceback
from collections import defaultdict
def getIndexDifference(idx_1, idx_2, count = 20):
    index_dict = defaultdict(list)
    idxDiff = set(idx_1).difference(set(idx_2))
    print(len(idxDiff))
    for game, action in idxDiff:
        index_dict[game].append(action)
    return index_dict
def getIdxDiff(db, idx):
    currentGame = None
    action_map = {}
    newIdxs = []
    for game_id, action_id in tqdm(idx):
        idx_old = (game_id, action_id)
        if game_id != currentGame:
            actions = db.actions(game_id = game_id)
            currentGame = game_id
            action_map = {idx : actions.loc[idx]["original_event_id"] for idx in actions.index}
        playId = int(float(action_map[idx_old]))
        idx_new = (game_id, playId)
        newIdxs.append(idx_new)
    return newIdxs
def main():
    errors = []
    oldFeats = pd.read_parquet("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/oldFeatures/all_features_defl_fail/y_scores_xg.parquet").index
    newFeats = pd.read_parquet("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_failed/y_scores_xg.parquet").index
    db = SQLiteDatabase("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql")
    #translatedIdxs = getIdxDiff(db, oldFeats)
    #index_dict = getIndexDifference(newFeats, translatedIdxs)
    index_dict = {"DFL-MAT-J03YJD": [18472900000008]}
    #with open('index_dict.json', 'r') as f:
    #    index_dict = json.load(f)
    for game_id in tqdm(list(index_dict.keys())):
        rdfPath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data"
        event, tracking = Animations.animationHelper(game_id, rdfPath)
        og_event = event.copy()
        og_tracking = tracking.copy()#some mutation problems - bandaid fix
        events = index_dict[game_id]
        events = list(map(float, events))
        intercepts = event[event["EVENT_ID"].isin(events)]
        for idx, row in tqdm(intercepts.iterrows()):
            try:
                event_id = row['EVENT_ID']
                frame = row['FRAME_NUMBER']
                custom_frames = range(int(frame), int(frame) + 75)
                title = f"Game {game_id} | Event {event_id}"
                animation = Animations.get_animation_from_raw(event_id, 1, og_event, og_tracking,custom_frames = custom_frames, show = False, add_frames = None, frameskip = 1, title = title)
                animation_title = f"playViz/{event_id}.gif"
                animation.save(animation_title, writer='pillow', fps=5, dpi=200)
            except:
                print(f"Error processing {game_id}, {event_id}, {traceback.format_exc()}")
if __name__ == "__main__":
    main()