#script which converts hard to read hawkeye json files to csv
import pandas as pd
import orjson
import regex as re 
import os
from tqdm import tqdm
def read_hawkeye_json(file_path):
    with open(file_path, 'rb') as f:
        first_line = f.readline()
    first_record = orjson.loads(first_line)#slowness from this json reading - not sure I can do much here
    player_dict = first_record['samples']['people']
    player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
    split = re.split("\.|\_", file_path.split("/")[-1])
    period = split[3]
    minute = split[4]
    if len(split) == 8:
        added = 0
    else:
        added = split[5]
    player_df['period'] = period
    player_df['minute'] = minute
    player_df['added'] = added
    player_df['time'] = [centroid[0]['time'] for centroid in player_df['centroid']]
    player_df['position'] = [centroid[0]['pos'] for centroid in player_df['centroid']]
    player_df['uefaId'] = [personId['uefaId'] if "uefaId" in personId else None for personId in player_df['personId']]
    player_df = player_df.dropna(subset=['uefaId']) # drops rows without uefaId
    player_df['speed'] = [centroid[0]['speed']['mps'] for centroid in player_df['centroid']] #get mps
    player_df['role'] = [role['name'] for role in player_df['role']]
    player_df = player_df[player_df['role'].isin(["Outfielder", "Goalkeeper"])]
    player_df['elapsed'] = (player_df['minute'].astype(float) + player_df['added'].astype(float) - 1) * 60 + player_df['time'].astype(float)
    player_df = player_df.sort_values(by = ['elapsed'], ascending = True)
    return player_df[['uefaId', 'time', 'position', 'speed', 'role', "period", "minute", "added", "elapsed"]]

def read_hawkeye_ball_json(file_path):
    with open(file_path, 'rb') as f:
        first_line = f.readline()
    first_record = orjson.loads(first_line)#slowness from this json reading - not sure I can do much here
    player_dict = first_record['samples']['ball']
    if len(player_dict) == 0:
        return pd.DataFrame()#empty if no record of ball
    player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
    split = re.split("\.|\_", file_path.split("/")[-1])
    period = split[3]
    minute = split[4]
    if len(split) == 8:
        added = 0
    else:
        added = split[5]
    player_df['period'] = period
    player_df['minute'] = minute
    player_df['added'] = added
    player_df['position'] = player_df['pos']
    player_df['speed'] = [centroid['mps'] for centroid in player_df['speed']] #get mps
    player_df['elapsed'] = (player_df['minute'].astype(float) + player_df['added'].astype(float) - 1) * 60 + player_df['time'].astype(float)
    player_df = player_df.sort_values(by = ['elapsed'], ascending = True)
    return player_df[['time', 'position', 'speed', "period", "minute", "added", "elapsed"]]
import traceback
def read_hawkeye_game(filepath, ball = False, output_dir = None):
    dfs = []
    print(filepath)
    minutes = [f"{filepath}/{file}" for file in os.listdir(filepath)]
    game = filepath.split("/")[8]
    iter = 1
    for minute in tqdm(minutes, leave = False):
        try:
            if ball:
                dfs.append(read_hawkeye_ball_json(minute))
            else:
                dfs.append(read_hawkeye_json(minute))
        except Exception as e:
            print(f"Error processing {minute}: {traceback.format_exc()}")
        iter += 1
    game_df = pd.concat(dfs)
    game_df = game_df.reset_index(drop=True)
    if output_dir is not None:
        game_df.to_csv(f"{output_dir}/{game}.csv", index = False)
    return game_df.sort_values(by = ["elapsed"], ascending=True)


path = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data"
centroids = "scrubbed.samples.centroids"
balls = "scrubbed.samples.ball"
output_dir_players = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs"
output_dir_ball = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_ball_csvs"
allplayers= [f"{path}/{file}/{centroids}" for file in os.listdir(path) if file[0] != "." and file not in ["tracking_csvs", "tracking_ball_csvs"]] # skip hidden files
allballs = [f"{path}/{file}/{balls}" for file in os.listdir(path) if file[0] != "." and file not in ["tracking_csvs", "tracking_ball_csvs"]]
def main(player, ball):
    if player:
        for file in tqdm(allplayers):
            read_hawkeye_game(file, False, output_dir_players)
    if ball:
        for file in tqdm(allballs):
            read_hawkeye_game(file, True, output_dir_ball)
main(False, True)