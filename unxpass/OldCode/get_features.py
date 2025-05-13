#Script to create features from db - legacy code
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
from unxpass import features as fs
from unxpass import labels as ls
warnings.filterwarnings(action="ignore", category=NoAuthWarning, module='statsbombpy')

allfeats = [f.__name__ for f in fs.all_features]
alllabels = [f.__name__ for f in ls.all_labels]
import time
import sqlite3
#"/home/lz80/rdf/sp161/shared/soccer-decision-making/buli_all.sql"
testpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/buli_50.sql"
DB_PATH = Path(testpath)
db = SQLiteDatabase(DB_PATH)
print(db.games().shape)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features_testing"#avg 10
from unxpass.datasets import PassesDataset, FailedPassesDataset, CompletedPassesDataset
print(allfeats)
dataset = PassesDataset(
        path=custom_path,
       xfns=["player_speed"],
       yfns = []
    )
dataset.create(db)
"""
unxpass train \
  /home/lz80/un-xPass/config \
  /home/lz80/rdf/sp161/shared/soccer-decision-making/m_euro_features_all \
  experiment="pass_success/threesixty"
"""