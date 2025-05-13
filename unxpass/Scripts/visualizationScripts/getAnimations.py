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
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.withSpeeds import pass_value_speeds
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.converters import playVisualizers
from unxpass.ratings_custom import LocationPredictions
from matplotlib.backends.backend_pdf import PdfPages
import mplsoccer
from unxpass.visualizers import plotPlays
from unxpass.Scripts.featureGenerators import Animations
from tqdm import tqdm
import json
import ast
def main():
    errors = []
    with open('index_dict.json', 'r') as f:
        index_dict = json.load(f)
    for game_id in tqdm(list(index_dict.keys())):
        event, tracking = visualizers_made.animationHelper(game_id)
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
                print("Error")
                event_id = row['EVENT_ID']
                errors.append(event_id)
if __name__ == "__main__":
    main()