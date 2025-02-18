#generates and saves gifs
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
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.ratings_custom import LocationPredictions

from matplotlib.backends.backend_pdf import PdfPages

from notebooks import playVisualizers
model_pass_value = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/20e7d3695d7049d0a513922d32b44a11/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive = False
)
DATA_DIR = Path("../stores/")
dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
feat_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_fail"
db = SQLiteDatabase(dbpath)
dataset_test = partial(PassesDataset, path = feat_path)
#surfaces = model_pass_value.predict_surface(dataset_test, db = db, model_name = "val", game_id = game_id)

from unxpass.load_xml import load_tracking, load_csv_event
from unxpass.visualizers_made import get_animation_from_raw
def get_trackingevent(game_id):
    trackingpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/{game_id}.xml"
    eventpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/KPI_Merged_all/KPI_MGD_{game_id}.csv"
    eventdf = load_csv_event(eventpath)
    trackingdf = load_tracking(trackingpath)
    return trackingdf, eventdf

def getgameiddata(game_id):
    labs = model_pass_value.initialize_dataset(dataset_test, model_name = "val").labels
    fails = labs[labs['concedes_xg'] > 0].index
    #db.actions(game_id = game_id).loc[fails]
    gameidx = [idx for idx in fails if idx[0] == game_id]
    eventids = db.actions(game_id = game_id).loc[gameidx]['original_event_id']
    trackingdf, eventdf = get_trackingevent(game_id)
    return eventids, trackingdf, eventdf
def save_gif(eventid, framerate, eventdf, trackingdf, frameskip, add_frames, filename):
    gif = get_animation_from_raw(float(eventid), framerate, eventdf, trackingdf, frameskip, False, add_frames)
    gif.save(filename)
framerate = 2
for game_id in db.games().index:
    eventids, trackingdf, eventdf = getgameiddata(game_id)
    for event in eventids:
        print(f"Animating event: {event}, game: {game_id}")
        filename = f"gifs/fail_xg_{game_id}_{event}.gif"
        gif = get_animation_from_raw(float(event), framerate, eventdf, trackingdf, frameskip = 20, show = False, add_frames = 600)
        gif.save(filename, writer='PillowWriter', fps=2)
#eventid
