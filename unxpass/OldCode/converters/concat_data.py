#Script which adds the new "1 second after" events to old events - legacy
import pandas as pd
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import re
sequences = "/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv"
sequences = pd.read_csv(sequences)
dir_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEyeResults_2"
dirfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
all_csvs = []
for file in dirfiles:
    print(f"Reading {file}")
    df = pd.read_csv(f"{dir_path}/{file}")
    all_csvs.append(df)
all_dfs = pd.concat(all_csvs)
#print(all_dfs['home_pass'])
print(len(sequences['id']))
all_dfs = all_dfs[all_dfs["home_pass"].astype(str).isin(sequences["id"].astype(str))]
print(all_dfs.shape[0])#should be 4257
all_dfs.to_csv(f"{dir_path}/hawkeye_second.csv")
missing = sequences[~sequences["id"].isin(all_dfs["home_pass"])][["BallReceipt", "id", "match_id"]]
print(missing[missing["BallReceipt"].notna()].shape[0])

