import os
import pandas as pd 
from pathlib import Path
def getOutlierIdx(speed_path):
    speeds = pd.read_parquet(speed_path)
    speeds['numoutlier'] = speeds.apply(lambda x: check_speeds(x['freeze_frame_360_a0']), axis = 1)
    return speeds[speeds['numoutlier'] > 0].index
def getSuccessIdx(success_path):
    success = pd.read_parquet(success_path)
    return success[success['success']].index
def check_speeds(frame):
    speed_list = [abs(player['x_velo']) > 11 or player['y_velo'] > 11 for player in frame]
    return sum(speed_list)
def getIdxs(dir_path, output_path, idxs, include = False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in  os.listdir(dir_path):
        print(f"Editing {file}")
        if include:
            parquet = pd.read_parquet(f"{dir_path}/{file}").loc[idxs]
        else:
            parquet = pd.read_parquet(f"{dir_path}/{file}")
            parquet = parquet[~parquet.index.isin(idxs)]
        parquet.to_parquet(f"{output_path}/{file}")
def main():
    input_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers"
    output_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_outliers_fail"
    successidx = getSuccessIdx(f"{input_dir}/y_success.parquet") #success idx
    getIdxs(input_dir, output_dir, idxs = successidx, include = False)
main()