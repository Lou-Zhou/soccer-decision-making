#Script to save surfaces in their raw dictionary form
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import json

import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset, SamePassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action_og
from unxpass.ratings_custom import LocationPredictions
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
STORES_FP = Path("../stores")
buli_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/buli_all.sql"
hawkeye_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
euro = "/home/lz80/rdf/sp161/shared/soccer-decision-making/euro_test.sql"
db = SQLiteDatabase(hawkeye_path)
db_og = SQLiteDatabase(euro)
db_buli = SQLiteDatabase(buli_path)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/m_euro_features_all"
#custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features"
dataset_test = partial(PassesDataset, path=custom_path)
component_id = "faa2f457ab094ac68dbf64b78f4da9c6"
model_pass_success_xg = load_model(f"runs:/{component_id}/component")
model_pass_success_sm = pass_success.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/7d3b32c51cde4a41abb7f4a319705086/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
#with open("euro_success_surfaces_xgb.json") as json_file:
#    xg_surfaces = json.load(json_file)
from matplotlib.backends.backend_pdf import PdfPages
import time
#print(xg_surfaces.keys())
#time.sleep(200)
idxs = list(model_pass_success_xg.initialize_dataset(dataset_test).features.index[0:300])
xg_surfaces = model_pass_success_xg.predict_surface(dataset_test, db = db_og, game_id = "3795108", action_id = idxs)
sm_surfaces = model_pass_success_sm.predict_surface(dataset_test, db = db_og, model_name = "val", game_id = "3795108")  
#xg_surfaces = model_pass_success_sm.predict_surface(dataset_test, db = db_og, model_name = "val", game_id = "3795108")
#surfaces = {"val":10}
# with open("euro_success_surfaces_xgb.json", "w") as outfile: 
#     json.dump(xg_surfaces, outfile, cls=NumpyEncoder)
# with open("euro_success_surfaces_xgb.json") as json_file:
#     xg_surfaces = json.load(json_file)

def save_plots_to_pdf(surface_1, surface_2, db, output_pdf_path):
    with PdfPages(output_pdf_path) as pdf:
        game_id = "3795108"
        for surface_name in surface_1:
            try:
                # Generate the plot
                action_id = int(surface_name.split("_")[1])
                print(game_id, action_id)
                SAMPLE = (game_id, action_id)
                ex_action = db.actions(game_id=SAMPLE[0]).loc[SAMPLE]
                surface_params = {"vmin": None, "vmax": None, "cmap": "Greens"}
                
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                fig.suptitle(f"Game ID: {game_id}, Action ID: {action_id}", fontsize=9)
                #print(surface[game_id][action_id])  # Optional: Debugging
                
                # Generate the plot for the current action
                plot_action_og(
                    ex_action, 
                    ax=axs[0], 
                    surface=surface_1[surface_name], 
                    surface_kwargs={"interpolation": "bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}
                )
                plot_action_og(
                    ex_action, 
                    ax=axs[1], 
                    surface=surface_2[game_id][action_id], 
                    surface_kwargs={"interpolation": "bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}
                )
                axs[0].title.set_text("Pass Success XGB")
                axs[0].title.set_fontsize(6)
                axs[1].title.set_text("Pass Success Soccermap")
                axs[1].title.set_fontsize(6)
                plt.tight_layout()
                
                # Save the current plot to the PDF
                pdf.savefig(fig)
                plt.close(fig)
            except: 
                print("error")
save_plots_to_pdf(xg_surfaces, sm_surfaces, db_og, "output_plots.pdf")
