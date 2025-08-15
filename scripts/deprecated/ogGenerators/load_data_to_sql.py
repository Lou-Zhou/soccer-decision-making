# Scripts which load raw json to sql db - legacy code

import warnings
import pandas as pd
import warnings
from typing import Any, Optional, cast
from functools import partial
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
from pandera.typing import DataFrame
from unxpass.components import pass_selection, pass_value, pass_success
from socceraction.spadl import config as spadlconfig
from socceraction.spadl.base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from socceraction.spadl.schema import SPADLSchema
import pandas as pd
import json
from pathlib import Path
from unxpass.databases import SQLiteDatabase
from unxpass.visualization import plot_action
from statsbombpy.api_client import NoAuthWarning
warnings.filterwarnings(action="ignore", category=NoAuthWarning, module='statsbombpy')

datasets = [
    { "getter": "local", "root": "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data", "competition_id":  9, "season_id": 281}
    #{ "getter": "local", "root": "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_subset", "competition_id":  9, "season_id": 281}
    #{ "getter": "local", "root": "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts", "competition_id":  53, "season_id": 106}
    ]#/home/lz80/rdf/sp161/shared/soccer-decision-making/match_plan/matches_DFL-SEA-0001K7.csv
#/home/lz80/rdf/sp161/shared/soccer-decision-making/converted_subset
#dataset = { "getter": "local", "root": "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data", "competition_id":  9, "season_id": 281}
#datasets = { "getter": "remote", "competition_id":  55, "season_id": 43}
import time
import sqlite3
DB_PATH = Path("/home/lz80/un-xPass/stores/buli_all.sql") 
db = SQLiteDatabase(DB_PATH)
games = pd.read_json("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/matches/9/281.json")
print(games['match_id'])
#for game in games['match_id'][0:50]:
#    print(f"Loading {game}")
#    dataset['game_id'] = game
#    db.import_data(**dataset)
for dataset in datasets:
    db.import_data(**dataset)
#DATA_DIR = Path("../stores/")
#from unxpass.datasets import PassesDataset, FailedPassesDataset, CompletedPassesDataset