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
model_pass_value_wChannel = pass_value.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/59e582d607844c31bb20b58578b90688/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive=  False
)

model_pass_value_nChannel = pass_value_custom.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/2808674780694efa899191b5bd0f7758/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    ), offensive=  False
)

#59e582d607844c31bb20b58578b90688
#No C,D: 20e7d3695d7049d0a513922d32b44a11
game_id = "3835331"
path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
db = SQLiteDatabase(path)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2"
dataset_test = partial(PassesDataset, path=custom_path)
receipts = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
ids = receipts[receipts['match_id'] == int(game_id)]['id']
surfaces_wchannel = model_pass_value_wChannel.predict_surface(dataset_test, db = db, model_name = "val", game_id = game_id)
surfaces_nchannel = model_pass_value_nChannel.predict_surface(dataset_test, db = db, model_name = "val", game_id = game_id)
pdf_filename = "buli_results_comparison_nolog.pdf"
actions = db.actions(game_id = game_id)
with PdfPages(pdf_filename) as pdf:
    for id in ids:
        print(id)
        id_new = f"{id}-1"
        game_id, action_id = actions[actions['original_event_id'] == id_new].index[0]
        fig = playVisualizers.plot_single_model_comparison(game_id, action_id, 
        f"Game Id: {game_id}, Statsbomb Id: {id}", surfaces_wchannel, surfaces_nchannel, db, subtitle_1 = "With Added Channels", subtitle_2 = "Without Added Channels", log = False)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

