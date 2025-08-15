# %%
#Script which converts receipts to passes for analysis - legacy code
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
import json
import os
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets_custom import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.ratings_custom import LocationPredictions

"""
{'id': 42, 'name': 'Ball Receipt*'}	
{'id': 30, 'name': 'Pass'}
"""

# %%
# Get the list of all files and directories
path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events_old/"
dir_list = os.listdir(path)

# %%
for match in dir_list:
    json_url = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events_old/"
    match_url = json_url + match
    output_url = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events_receipt/"
    output_path = output_url + match
    test_json = pd.read_json(match_url,convert_dates = False)
    #can we just do an empty pass - see what happens
    test_json['type'] = np.where(test_json["type"] == {'id': 30, 'name': 'Pass'}, {'id': -1, 'name': 'Old_Pass'}, test_json['type'])
    test_json['type'] = np.where(test_json["type"] == {'id': 42, 'name': 'Ball Receipt*'},{'id': 30, 'name': 'Pass'}, test_json['type'])
    test_json['pass'] = test_json.apply(lambda d: {"end_location":d["location"], 'body_part':{'id':40, 'name':'Right Foot'}} if d['type'] == {'id': 30, 'name': 'Pass'} else None, axis = 1)
    json_str = test_json.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
        #cleaned_data = remove_nan(json_data)
    print(f"Writing {match}")
    # Save to a file
    with open(output_path, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=2)


