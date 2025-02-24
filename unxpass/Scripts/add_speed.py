from unxpass import load_xml
import pandas as pd
import numpy as np
import json
from unxpass.databases import SQLiteDatabase

def getFlips(game_id):
    events = load_xml.load_event(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/event_data_all/{game_id}.xml")
    events[['TeamLeft', 'TeamRight']] = events[['TeamLeft', 'TeamRight']].fillna(method='ffill')
    #direction of play to right
    #if team in possession is team left - all good
    #if team in possession is team right - need to flip
    return events.loc[events['Team'] == events['TeamRight'], 'EventId']
def frametodict(group, shouldFlip):
    """
    Converts a group of tracking data into a dictionary with player IDs as keys and their translated positions.
    """
    # Exclude the BALL rows and work on a copy to avoid SettingWithCopyWarning
    noball = group[group['TeamId'] != "BALL"].copy()

    # Ensure numeric conversion for X and Y (if not already floats)
    noball['X'] = noball['X'].astype(float)
    noball['Y'] = noball['Y'].astype(float)
    
    # Compute translated coordinates using vectorized operations
    noball['X_translated'] = 120 - (1.09361 * noball['X'] + 60)
    noball['Y_translated'] = 80 - (-1.09361 * noball['Y'] + 40)
    
    # Apply flip if necessary
    if shouldFlip:
        noball['X_translated'] = 120 - noball['X_translated']
        noball['Y_translated'] = 80 - noball['Y_translated']
    
    # Build the dictionary using itertuples for faster iteration
    locs = {
        row.PersonId: {"X": row.X_translated, "Y": row.Y_translated, "Team": row.TeamId}
        for row in noball.itertuples(index=False)
    }
    
    return locs



#getting speeds from n prior frames for buli data

#load all relevant data
def getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, id_to_event, flips, framesback = 10):
    timediff = 0.04 * framesback
    buli_id = id_to_event[(game_id, action_id)]
    shouldflip = buli_id in flips
    
    event = eventcsv[eventcsv['EVENT_ID'] == int(buli_id.split(".")[0])].iloc[0]
    event_frame = event['FRAME_NUMBER']
    team = event["CUID1"]
    event_frame_str = str(event_frame)
    prior_frame_str = str(event_frame - framesback)
    try:
        event_pos = tracking_groups.get_group(event_frame_str)
    except KeyError:
        event_pos = pd.DataFrame()
    try:
        prior_pos = tracking_groups.get_group(prior_frame_str)
    except KeyError:
        prior_pos = pd.DataFrame()
    if len(event_pos) > 0:
        event_pos = frametodict(event_pos, shouldflip)
    if len(prior_pos) > 0:
        prior_pos = frametodict(prior_pos, shouldflip)
    #return event_pos, prior_pos
    speed_output = []
    for player, pos in event_pos.items():
        isTeammate = event_pos[player]["Team"] == team
        prior = prior_pos.get(player)
        if player not in prior_pos:
            x_diff = 0
            y_diff = 0
        else:
            x_diff = event_pos[player]["X"] - prior_pos[player]["X"]
            y_diff = event_pos[player]["Y"] - prior_pos[player]["Y"]
        x_velo = x_diff / timediff
        y_velo = y_diff / timediff
        player_dict = {
            "player": player,
            "isTeammate": isTeammate,
            "x_velo": x_velo,
            "y_velo": y_velo,
            "location": [event_pos[player]["X"], event_pos[player]["Y"]],
        }
        speed_output.append(player_dict)
    return speed_output
    #need is teammate, location, and speed 

def addAllSpeedBuli(skeleton, db):
    games = skeleton.index.get_level_values(0).unique()
    output = pd.DataFrame(index = skeleton.index)
    output["speed_frame_360_a0"] = np.nan
    output["speed_frame_360_a0"] = output["speed_frame_360_a0"].astype(object)
    iter = 1
    for game_id in games:
        id_to_event = db.actions(game_id = game_id)['original_event_id'].to_dict()
        eventcsv = load_xml.load_csv_event(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/KPI_Merged_all/KPI_MGD_{game_id}.csv")
        trackingdf = load_xml.load_tracking(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/{game_id}.xml")
        tracking_groups = trackingdf.groupby('N')
        game_mask = skeleton.index.get_level_values(0) == game_id
        flips = getFlips(game_id)
        for game_id, action_id in skeleton.index[game_mask]:
            print(f"Processing match {game_id}, action {action_id}, {iter}/{len(games)}", end = "\r")
            output.at[(game_id, action_id), "speed_frame_360_a0"] = [getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, id_to_event, flips = flips)]
        iter += 1
    return output


#HAWKEYE FUNCTIONS
def getGksTM(game_id):
        lineups = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/lineups/{game_id}.json"
        lineup_df = pd.read_json(lineups, convert_dates = False)
        team_1 = lineup_df['team_id'].loc[0]
        team_2 = lineup_df['team_id'].loc[1]
        team_1_dict = lineup_df['lineup'].loc[0]
        team_2_dict = lineup_df['lineup'].loc[1]
        team_1_lineup = [player_dict['player_id'] for player_dict in team_1_dict]
        team_2_lineup = [player_dict['player_id'] for player_dict in team_2_dict]
        team_map = {team_1 : team_1_lineup, team_2 : team_2_lineup}#building a map of team id to player ids
        player_to_team = {player_id: team_id for team_id, players in team_map.items() for player_id in players} #mapping players to teams
        pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_1_dict if len(player['positions']) > 0}
        team_2_pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_2_dict if len(player['positions']) > 0}
        pos_dict.update(team_2_pos_dict)
        goalkeepers = [key for (key,value) in pos_dict.items() if value == "Goalkeeper"]
        return player_to_team, goalkeepers#gets set of goalkeepers too
def clean_dict(freeze_frame):
    output = {}
    for player in freeze_frame:
        player_dict = {}
        player_dict['teammate'] = player['teammate']
        player_dict['x'] = player['location'][0]
        player_dict['y'] = player['location'][1]
        output[player['player']] = player_dict
    return output
def getSpeedsHawkeye(init, event, timediff):
    output = []
    for player in event:
        player_dict = {}
        if player not in init:
            continue
        x_diff = event[player]['x'] - init[player]['x']
        y_diff = event[player]['y'] - init[player]['y']
        x_velo = x_diff / timediff
        y_velo = y_diff / timediff
        player_dict['player'] = player
        player_dict['teammate'] = event[player]['teammate']
        player_dict['x_velo'] = x_velo
        player_dict['y_velo'] = y_velo
        player_dict['location'] = [event[player]['x'], event[player]['y']]
        output.append(player_dict)
    return output

def getSpeedHawkeye(match_id, action_id, tracking, player_to_team, goalkeepers, id_to_event):
    full_id = id_to_event[(str(match_id), action_id)]
    if len(full_id.split("-")) != 6:
        return {"empty": "empty"}
    real_id = full_id.rsplit("-", 1)[0]
    frame_idx = int(full_id.rsplit("-",1)[1])
    pass_data = sequences[sequences['id'] == real_id].iloc[0]
    timeback = 0.4
    time = pass_data['BallReceipt']
    period = pass_data['period']
    minute = int((time + .04 * int(frame_idx)) / 60 + 1)
    second = int((time + .04 * int(frame_idx)) % 60)
    second_range = (second - timeback, second + .01)
    team = pass_data['team_id']
    actor =  pass_data['player_id']
    uefa_map = {}
    file_path_begin = os.listdir(tracking)[0].rsplit("_", 2)[0]
    #goalkeepers = []
    file_path = f"{tracking}/{file_path_begin}_{str(period)}_{str(minute)}.football.samples.centroids"
    all_locs = []
    loc1 = conversions.read_Hawkeye_player_loc(file_path, period, minute, second_range, team,actor, real_id, player_to_team, goalkeepers)
    all_locs.append(loc1)
    if(second - timeback < 0):
        #if the time is negative, we need to get the last frame of the previous minute
        file_path = f"{tracking}/{file_path_begin}_{str(period)}_{str(minute - 1)}.football.samples.centroids"
        second_range = (59 + second - timeback, 60)
        loc2 = conversions.read_Hawkeye_player_loc(file_path, period, minute - 1, second_range, team,actor, real_id, player_to_team, goalkeepers)
        all_locs.append(loc2)
    addedtracking = pd.concat(all_locs)
    addedtracking['event_uuid'] = [f"{real_id}-b{i}" for i in range(len(addedtracking), 0, -1)]
    last = clean_dict(addedtracking.loc[0]["freeze_frame"])
    first = clean_dict(addedtracking.loc[len(addedtracking) - 1]["freeze_frame"])
    return getSpeedsHawkeye(first, last, timeback)
def getAllHawkeyeSpeeds(skeleton, hawkeye_db):
    with open("/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    output = pd.DataFrame(index = skeleton.index)
    output["speed_frame_360_a0"] = np.nan
    output["speed_frame_360_a0"] = output["speed_frame_360_a0"].astype(object)
    sb_to_hawkeye = {v: k for k, v in hawkeye_to_sb.items()}
    iter = 1
    for match_id in skeleton.index.get_level_values(0).unique():

        hawkeye_id = sb_to_hawkeye[int(match_id)]
        sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
        tracking = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/{hawkeye_id}/scrubbed.samples.centroids"
        output["speed_frame_360_a0"] = {}
        id_to_event = hawkeye_db.actions(game_id = match_id)['original_event_id'].to_dict()
        player_to_team, goalkeepers = getGksTM(match_id)
        game_mask = skeleton.index.get_level_values(0) == match_id
        for game_id, action_id in skeleton.index[game_mask]:
            print(f"Processing match {match_id}, action {action_id}, {iter}/{len(skeleton.index.get_level_values(0).unique())}", end = "\r")
            #get player to team mapping
            output.at[(game_id, action_id), "speed_frame_360_a0"] = [getSpeedHawkeye(game_id, action_id, tracking, player_to_team, goalkeepers, id_to_event)]
        iter += 1
    return output
#getAllHawkeyeSpeeds(match_id, hawkeye_db)

#RUN BULI
print("Adding Bundesliga Speeds...")
skeleton = pd.read_parquet("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_fixed/x_endlocation.parquet")
dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
feat_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features_fixed"
db = SQLiteDatabase(dbpath)
buli_speed = addAllSpeedBuli(skeleton, db)
buli_speed.to_parquet(f"{feat_path}/x_speed_frame_360.parquet")

#RUN HAWKEYE
print("Adding Hawkeye Speeds...")
sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
feat_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2"
skeleton = pd.read_parquet(f"{feat_path}/x_endlocation.parquet")
dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
hawkeye_db = SQLiteDatabase(dbpath)
hawkeye_speed = getAllHawkeyeSpeeds(skeleton, hawkeye_db)
hawkeye_speed.to_parquet(f"{feat_path}/x_speed_frame_360.parquet")