import pandas as pd
import json
import os
from unxpass.Scripts.featureGenerators import getHawkeyeFeats
from unxpass.databases import SQLiteDatabase
from tqdm import tqdm
import numpy as np
import traceback 
def getHeSpeed(tracking_folder, skeleton_path, dbpath, framesback, framesforward, ball = False):
    """
    Function which generates for all hawkeye data from a db implementation(originally in un-xpass)
    """
    player_speeds = []
    ball_starts = []
    ball_speeds = []
    sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequences_new.csv")
    timeelapsed = {
    1:0,
    2:45 * 60,
    3: 90 * 60,
    4: 105 * 60
    }
    sequences["BallReceipt"] = sequences["period"].map(timeelapsed) + sequences["BallReceipt"]#minute adjustment
    with open("../../../../rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    skeleton = pd.read_parquet(skeleton_path).index
    hawkeye_db = SQLiteDatabase(dbpath)
    framesback = 5
    framesforward = 5
    alltracking = [file for file in os.listdir(tracking_folder) if file.endswith(".csv")]
    uefa_map = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/player_ids_matched.csv")
    uefa_map = pd.Series(uefa_map["sb_player_id"].values,index=uefa_map["uefa_player_id"]).to_dict()
    for game_file in tqdm(alltracking):
    #for game_file in test_track:
        if ball:
            all_dfs = getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, hawkeye_db, framesback, framesforward, sequences, ball)
            player_speeds.append(all_dfs[0])
            ball_starts.append(all_dfs[1])
            ball_speeds.append(all_dfs[2])
        else:
            player_speeds.append(getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, hawkeye_db, framesback, framesforward, sequences, ball))
    player_speed_df = pd.concat(player_speeds)
    if ball:
        ball_start_df = pd.concat(ball_starts)
        ball_speeds_df = pd.concat(ball_speeds)
        return player_speed_df, ball_start_df, ball_speeds_df
    #return speed_df
    return player_speed_df
def getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, db, framesback, framesforward, sequences, ball = False):
    """
    Gets hawkeye features for an entire game
    """
    #game_file = "2032219_Portugal_Switzerland.csv"
    game = game_file.split(".")[0]
    
    tracking_path = f"../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs/{game_file}"
    
    tracking = pd.read_csv(tracking_path)#.sort_values(by = ["elapsed"])
    tracking['statsbombid'] = tracking['uefaId'].astype(int).map(uefa_map)
#3835338, action 7218
    statsbomb_id = hawkeye_to_sb[game]

    game_mask = skeleton.get_level_values(0) == str(statsbomb_id)
    game_skeleton = skeleton[game_mask]
    team_dict, gks = getHawkeyeFeats.getGksTM(statsbomb_id)
    tracking['team'] = tracking['statsbombid'].map(team_dict)
    tracking['isGk'] = tracking['role'] == "Goalkeeper"
    speed_df = pd.DataFrame(index = game_skeleton)
    action_df = db.actions(game_id = int(statsbomb_id))
    action_map = pd.Series(action_df['original_event_id'].values, index=action_df.index).to_dict()
    player_speeds = pd.DataFrame(index = game_skeleton)
    if ball:
        ball_tracking_path = f"/../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_ball_csvs/{game_file}"
        ball_df = pd.read_csv(ball_tracking_path)
        ball_speeds = pd.DataFrame(index = game_skeleton)
        ball_starts = pd.DataFrame(index = game_skeleton)
    player_speeds["freeze_frame_360_a0"] = np.nan
    player_speeds["freeze_frame_360_a0"] = player_speeds["freeze_frame_360_a0"].astype(object) #enforce some consistency
    for game_id, action_id in tqdm(game_skeleton, leave = False):
        #action_id = 7597
        #pbar.set_description(f"Processing game {game_id}, action {action_id}")
        action_sb_id = action_map.get((game_id, action_id))
        if len(action_sb_id.split("-")) == 5:#if non-interesting event
            sb_action_id = action_sb_id
            frame_idx = 0
        else:
            sb_action_id = action_sb_id.rsplit("-", 1)[0]#if interesting event(denoted by dash)
            frame_idx = action_sb_id.rsplit("-", 1)[1]
        
           
        try:
            if ball:
                speed_dict, ball_dict = getHawkeyeFeats.he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball, ball_df)
                
                ball_starts.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                ball_starts.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
            else:
                speed_dict = getHawkeyeFeats.he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball)
            player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
        except Exception as e:
            res = dict((v,k) for k,v in hawkeye_to_sb.items())
            print(f"Error processing game {game_id}, {res[int(game_id)]}, action {action_id}: {traceback.format_exc()}")
            speed_dict = {}
    if ball:
        return player_speeds, ball_starts, ball_speeds
    return player_speeds
def main(ball):
    #generate from pregenerated statsbomb features - legacy code
    dbpath = "../../../../rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
    tracking_folder = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs"
    skeleton_path = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_endlocation.parquet"
    output_path = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_frame_360.parquet"
    #player_speed_df.loc[skeleton], ball_start_df.loc[skeleton], ball_speeds_df.loc[skeleton]
    if ball:
        all_dfs = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
        ball_start_output = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_startlocation_3.parquet"
        ball_speed_output = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_3.parquet"
        speeddf = all_dfs[0]
        all_dfs[1].to_parquet(ball_start_output)
        all_dfs[2].to_parquet(ball_speed_output)
    #print(speeddf)
    else:
        speeddf = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
    speeddf.to_parquet(output_path)
if __name__ == "__main__": main(True)