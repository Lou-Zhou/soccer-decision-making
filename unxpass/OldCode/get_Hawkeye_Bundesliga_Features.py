"""generates features from scratch for bundesliga and hawkeye data 
legacy code - combined hawkeye and bundesliga code for reference sake, 
should just run the buli or hawkeye code by itself
"""
from unxpass import load_xml
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from unxpass.converters import conversions
from unxpass.databases import SQLiteDatabase
import traceback

def getFlips(game_id):
    events = load_xml.load_event(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/event_data_all/{game_id}.xml")
    events[['TeamLeft', 'TeamRight']] = events[['TeamLeft', 'TeamRight']].fillna(method='ffill')
    #direction of play to right
    #if team in possession is team left - all good
    #if team in possession is team right - need to flip
    return events.loc[events['Team'] == events['TeamRight'], 'EventId']
def frametodict(group, shouldFlip, ball = False):
    """
    Converts a group of tracking data into a dictionary with player IDs as keys and their translated positions.
    """
    # Exclude the BALL rows and work on a copy to avoid SettingWithCopyWarning
    tracking = group.copy()

    # Ensure numeric conversion for X and Y (if not already floats)
    tracking['X'] = tracking['X'].astype(float)
    tracking['Y'] = tracking['Y'].astype(float)
    
    # Compute translated coordinates using vectorized operations
    tracking['X_translated'] = (tracking['X'] + 105/2)
    tracking['Y_translated'] = 68 - (tracking['Y'] + 34)#features are already in meters, no need to convert
    
    # Apply flip if necessary - dont think its necessary since features set is already flipped(?)
    #if shouldFlip:
    #    noball['X_translated'] = 105 - noball['X_translated']
    
    # Build the dictionary using itertuples for faster iteration
    locs = {
        row.PersonId: {"X": row.X_translated, "Y": row.Y_translated, "Team": row.TeamId}
        for row in tracking.itertuples(index=False)
    }
    
    return locs

def getIntercept(angles, row):
    start_frame = row['FRAME_NUMBER'] + 2
    end_frame = row['FRAME_NUMBER'] + 2 + (2 / .04)
    angleFrames = angles[(angles['N'] >= start_frame) & (angles['N'] <= end_frame)]
    outOfBounds = (abs(angleFrames['X'].astype(float)) > 52.5) | (abs(angleFrames['Y'].astype(float)) > 34)
    endMask = (angleFrames['angle_change'] > 20) | (outOfBounds)
    angleFrames = angleFrames[endMask]
    if len(angleFrames) == 0:
        return None
    return angleFrames.loc[angleFrames['N'].idxmin()]['N']


def getAngleFrames(sample_pass, allAngles):
    start_frame = sample_pass['FRAME_NUMBER']
    event_id = sample_pass['EVENT_ID']
    outOfBounds = (abs(allAngles['X'].astype(float)) > 52.5) | (abs(allAngles['Y'].astype(float)) > 34)
    angleStartFrame = allAngles[((allAngles['angle_change'] > 20) & (allAngles['N'] > start_frame + 10))  | ((outOfBounds) & (allAngles['N'] > start_frame + 5)) ]
    if len(angleStartFrame) == 0:
        return start_frame + (2 / .04) #get 2 seconds after pass
    angleStartFrame = angleStartFrame.loc[angleStartFrame['N'].idxmin()]['N']
    return angleStartFrame
#visualize_coords_from_tracking()
def getAngleChangeFromStart(tracking, pass_start_map, fb = 10):
    """
    Compute angle change between ball direction at pass start and at each frame.

    Parameters:
    - tracking: DataFrame with ball tracking data (must contain 'N', 'X', 'Y', 'GameSection', 'TeamId')
    - pass_start_map: dict or Series mapping frame N to pass start frame N0

    Returns:
    - DataFrame with X, Y, angle_change, N, and GameSection
    """
    ball_data_df = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball_data_df['N'] = ball_data_df['N'].astype(float)
    ball_data_df['X'] = ball_data_df['X'].astype(float)
    ball_data_df['Y'] = ball_data_df['Y'].astype(float)
    ball_data_df = ball_data_df.sort_values(by='N')

    # Compute dx, dy for current frame
    ball_data_df['dx'] = ball_data_df['X'] - ball_data_df.groupby('GameSection')['X'].shift(10)
    ball_data_df['dy'] = ball_data_df['Y'] - ball_data_df.groupby('GameSection')['Y'].shift(10)
    ball_data_df['magnitude'] = np.sqrt(ball_data_df['dx']**2 + ball_data_df['dy']**2)

    # Map each N to its pass start frame
    ball_data_df['pass_start_N'] = ball_data_df['N'].map(pass_start_map)
    
    ball = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball[['N', 'X', 'Y']] = ball[['N', 'X', 'Y']].astype(float)
    ball = ball.sort_values('N')
    ball = ball.drop_duplicates(subset='N')  # Ensure no duplicate frames
    ball_indexed = ball.set_index('N')

    # Get start and end positions in one go
    start_pos = ball_indexed.reindex(ball_data_df['pass_start_N'])
    end_pos = ball_indexed.reindex(ball_data_df['pass_start_N'] + fb)

    # Compute vector difference
    start_dxdy = end_pos[['X', 'Y']].values - start_pos[['X', 'Y']].values

    # Assign to new columns
    ball_data_df[['start_dx', 'start_dy']] = start_dxdy
    ball_data_df['start_magnitude'] = np.hypot(ball_data_df['start_dx'], ball_data_df['start_dy'])
    
    # Compute dot product between current and pass start vectors
    ball_data_df['dot_product'] = (ball_data_df['dx'] * ball_data_df['start_dx'] +
                                   ball_data_df['dy'] * ball_data_df['start_dy'])

    # Calculate angle in degrees
    cosine_values = ball_data_df['dot_product'] / (ball_data_df['magnitude'] * ball_data_df['start_magnitude'])
    cosine_values = np.clip(cosine_values, -1, 1)
    ball_data_df['angle_change'] = np.degrees(np.arccos(cosine_values))

    # Clean up infinities and NaNs
    ball_data_df['angle_change'].replace([np.inf, -np.inf], np.nan, inplace=True)
    ball_data_df['angle_change'].fillna(0, inplace=True)

    return ball_data_df[['X', 'Y', 'angle_change', 'N','pass_start_N', 'GameSection', 'start_dx', 'start_dy', 'start_magnitude']]
#load all relevant data
import regex as re
def getClosestFrame(row, tracking, angle):
    if row["SUBTYPE"] != "Pass" and row['SUBTYPE'] != "Cross":
        return row["RECFRM"]#if not pass / cross - essentiallly None
    if not pd.isna(row["RECFRM"]) and abs(row["RECFRM"] - row["FRAME_NUMBER"]) > 5:
        return row["RECFRM"]#if already has receipt frame from event data, and recfrm is reasonable
    #if we don't know recipient or the evaluation is false - use 20 degree method    
    if pd.isna(row["PUID2"]) or row["EVALUATION"] == "unsuccessful":
        angleStartFrame = getAngleFrames(row, angle)
        if abs(angleStartFrame - row["FRAME_NUMBER"]) < (2 / .04):
            return angleStartFrame#check if within 2 seconds
        if pd.isna(row["PUID2"]):
            return row["FRAME_NUMBER"] + (2/.04) #if no recipient or angle frame, get 2nd frame

    #if we know recipient and successful, find time when recipient is closest:
    startFrame = float(row["FRAME_NUMBER"])
    endFrame =  float(row["NEXT_FRAME"])
    if pd.isna(endFrame):#very specific edge case where last pass is also no receiver and no other events
        endFrame = startFrame + 80
    if abs(startFrame - endFrame) < 5:
        endFrame = endFrame + 80  # Adjust endFrame if too close
    player = row["PUID2"]
    
    between_condition = f"N > {startFrame} and N < {endFrame}"
    player_query = f"{between_condition} and PersonId == @player"
    ball_query = f"{between_condition} and TeamId == 'BALL'"
    
    player_locs = tracking.query(player_query)[["N", "X", "Y"]].rename(columns={"X": "Player_X", "Y": "Player_Y"})
    ball_locs = tracking.query(ball_query)[["N", "X", "Y"]].rename(columns={"X": "Ball_X", "Y": "Ball_Y"})

    all_locs = player_locs.merge(ball_locs, on="N")#merge player and ball 
    
    if all_locs.empty:
        return float(row["NEXT_FRAME"])
    
    # Calculate squared distance
    all_locs["DistanceSq"] = (all_locs["Player_X"] - all_locs["Ball_X"])**2 + (all_locs["Player_Y"] - all_locs["Ball_Y"])**2
    
    # Get frame with minimum squared distance
    frame = all_locs.loc[all_locs["DistanceSq"].idxmin()]["N"]
    return frame
def map_n_event(df1, df2):
    #maps frames to events, closest to event frame while still greater
    df1 = df1.copy()
    df2 = df2.copy()
    df1['N'] = df1['N'].astype(int)
    df2['FRAME_NUMBER'] = df2['FRAME_NUMBER'].astype(int)
    frame_vals = np.sort(df2['FRAME_NUMBER'].unique())

    # Extract N values
    n_values = df1['N'].values

    # Find insertion index for each N using 'left' (ceiling search)
    indices = np.searchsorted(frame_vals, n_values, side='right') - 1

    # For invalid indices, set to 0 (use the smallest FRAME_NUMBER)
    indices[indices < 0] = 0

    # Build mapping: all N map to a valid FRAME_NUMBER
    mapped_frames = frame_vals[indices]
    return dict(zip(n_values, mapped_frames + 1))




def getReceipts(tracking, eventdf):
    #gets receipt frames
    eventdf = eventdf.sort_values(by=["FRAME_NUMBER", "RECFRM"], na_position="last")
    n_map = map_n_event(tracking, eventdf)#generates mapping between frames and events
    angles = getAngleChangeFromStart(tracking, n_map) #gets the change in angle between the start of a pass and end
    eventdf['RECFRM'] = eventdf.groupby('FRAME_NUMBER')['RECFRM'].transform(
    lambda x: x.fillna(x.dropna().iloc[0]) if not x.dropna().empty else x
        )#replaces non na values in group with first non na value
    eventdf["NEXT_FRAME"] = eventdf["FRAME_NUMBER"].shift(-1).fillna(method='bfill')
    eventdf.loc[eventdf["NEXT_FRAME"] == eventdf["FRAME_NUMBER"], "NEXT_FRAME"] = np.nan
    eventdf["NEXT_FRAME"] = eventdf["NEXT_FRAME"].fillna(method='bfill')#is this necessary?
    #if the difference between two events is over 300, then default to the frame(practically only occurs during halftime and fulltime whistles, or with missing events)
    eventdf.loc[abs(eventdf["NEXT_FRAME"] - eventdf["FRAME_NUMBER"]) > 300, "NEXT_FRAME"] = eventdf.loc[abs(eventdf["NEXT_FRAME"] - eventdf["FRAME_NUMBER"]) > 300, "FRAME_NUMBER"]
    tracking["N"] = tracking["N"].astype(float)
    tracking["X"] = tracking["X"].astype(float)
    tracking["Y"] = tracking["Y"].astype(float)
    #gets closest frame
    eventdf["RECFRM"] = eventdf.apply(lambda row: getClosestFrame(row, tracking, angles), axis = 1)
    mask = eventdf['RECFRM'] == eventdf['FRAME_NUMBER']#this should be redundant, but just in case - check for out of bounds
    eventdf.loc[mask, "RECFRM"] = eventdf.loc[mask].apply(lambda row: getOutOfBounds(row, tracking), axis = 1)
    return eventdf
def getOutOfBounds(row, tracking):
    outOfBounds = (abs(tracking['X']) > 52.5) | (abs(tracking['Y']) > 34)
    start_frame = row["FRAME_NUMBER"]
    end_frame = row['RECFRM'] + 300
    ball_data = tracking[(tracking["N"] >= start_frame) & (tracking["N"] <= end_frame) & (tracking["TeamId"] == "BALL") & (outOfBounds)]
    if len(ball_data) > 0:
        return min(ball_data["N"])
    else:
        return row['RECFRM']
def getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, border, flips, gks, framesback = 5, framesforward = 5, ball = False):
    timediff = 0.04 * (framesback + framesforward)
    buli_id = action_id
    shouldflip = buli_id in flips
    event = eventcsv[eventcsv['EVENT_ID'] == buli_id].iloc[0]
    
    
    event_frame = event['FRAME_NUMBER']
    team = event["CUID1"]
    actor = event["PUID1"]
    if pd.isna(team) or pd.isna(actor):
        team = event["CUID2"]
        actor = event["PUID2"]
    if pd.isna(event['PUID2']):
        recipient = None
    else:
        recipient = event['PUID2']
    current_pos = tracking_groups.get_group(event_frame)
    period = current_pos["GameSection"].iloc[0]
    period_start = int(border["first"][period])
    period_end = int(border["last"][period])
    
    endFrame = event["RECFRM"]
    event_frame_str = min(event_frame + framesforward, period_end)
    prior_frame_str = max(event_frame - framesback, period_start)
    end_frame_str = min(endFrame, period_end)
    #edge case of half time occuring
    event_pos = tracking_groups.get_group(event_frame_str)
    prior_pos = tracking_groups.get_group(prior_frame_str)
    
    if len(event_pos) > 0:
        event_pos = frametodict(event_pos, shouldflip, ball)
    if len(prior_pos) > 0:
        prior_pos = frametodict(prior_pos, shouldflip, ball)
    if len(current_pos) > 0:
        current_pos = frametodict(current_pos, shouldflip, ball)
    ball_ff = None
    if ball:
        end_pos = tracking_groups.get_group(end_frame_str)
        end_pos = frametodict(end_pos, shouldflip, ball)["DFL-OBJ-0000XT"]
        event_pos_ball = event_pos["DFL-OBJ-0000XT"]
        prior_pos_ball = prior_pos["DFL-OBJ-0000XT"]
        current_pos_ball = current_pos["DFL-OBJ-0000XT"]
        x_velo = (event_pos_ball["X"] - prior_pos_ball["X"]) / timediff
        y_velo = (event_pos_ball["Y"] - prior_pos_ball["Y"]) / timediff
        ball_ff = {
            "start_x": current_pos_ball["X"],
            "start_y": current_pos_ball["Y"],
            "speed_x": x_velo,
            "speed_y": y_velo,
            "end_x": end_pos["X"],
            "end_y":end_pos["Y"]
        }
    #return event_pos, prior_pos
    speed_output = []
    for player, pos in event_pos.items():
        if player != "DFL-OBJ-0000XT":
            isTeammate = event_pos[player]["Team"] == team
            isActor = actor == player
            #print(isActor)
            prior = prior_pos.get(player)
            if player not in prior_pos:
                x_diff = 0#I think this is redundant
                y_diff = 0
            else:
                x_diff = event_pos[player]["X"] - prior_pos[player]["X"]
                y_diff = event_pos[player]["Y"] - prior_pos[player]["Y"]
            x_velo = x_diff / timediff
            y_velo = y_diff / timediff
            isRecipient = recipient == player
            isGoalKeeper = player in gks.values
            player_dict = {
                "player": player,
                "actor":isActor,
                "teammate": isTeammate,
                "x_velo": x_velo,
                "y_velo": y_velo,
                "x": current_pos[player]["X"], #player is in event_pos, but not current_pos?
                "y": current_pos[player]["Y"],
                "goalkeeper":isGoalKeeper,
                "recipient": isRecipient
            }
            speed_output.append(player_dict)
    return getFlip(speed_output, ball_ff)
def checkDeadBall(row, eventcsv):
    deadBalls = ['Kickoff', 'ThrowIn', 'FreeKick','GoalKick']
    if row['SUBTYPE'] not in ['Pass', 'Cross']:#ensure is a legitimate pass
        return -1
    if (eventcsv['FRAME_NUMBER'] == row['FRAME_NUMBER']).sum() > 1:
        framecsv = eventcsv[eventcsv['FRAME_NUMBER'] == row['FRAME_NUMBER']]
        if framecsv['SUBTYPE'].isin(deadBalls).any():
            return -1#ensure also not a deadball situation
    return 1
def addAllSpeedBuli(games, ball = False):
    #change db here
    index = pd.MultiIndex.from_tuples([], names=['game_id', 'action_id'])
    player_speeds = pd.DataFrame(columns = ['freeze_frame_360_a0'], index = index)
    
    player_speeds["freeze_frame_360_a0"] = np.nan
    player_speeds["freeze_frame_360_a0"] = player_speeds["freeze_frame_360_a0"].astype(object)
    if ball:
        ball_speeds = pd.DataFrame(columns = ["speedx_a02", "speedy_a02"], index = index)
        ball_start = pd.DataFrame(columns = ["start_x_a0", "start_y_a0"], index = index)
        ball_end = pd.DataFrame(columns = ["end_x_a0", "end_y_a0"], index = index)
    iter = 1
    for game_id in tqdm(games):
        try:
            #game_id = 'DFL-MAT-J03YFX'
            #id_to_event = db.actions(game_id = game_id)['original_event_id'].to_dict()
            eventcsv = load_xml.load_csv_event(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
            trackingdf = load_xml.load_tracking(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/zipped_tracking/zip_output/{game_id}.xml")
            lineups = load_xml.load_players(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/match_information/{game_id}.xml", False)
            first_frame = {}
            last_frame = {}
            eventcsv = getReceipts(trackingdf, eventcsv)
            gks =  lineups[lineups['PlayingPosition'] == "TW"]["PersonId"]
            #none of these games should go to extra time
            period = trackingdf["GameSection"].unique()[0]
            test = trackingdf[trackingdf["GameSection"] == period]["N"]
            for period in trackingdf["GameSection"].unique():#probably can be optimized by a groupBy
                first_frame[period] = trackingdf[trackingdf["GameSection"] == period]["N"].astype(int).min()
                last_frame[period] = trackingdf[trackingdf["GameSection"] == period]["N"].astype(int).max()
            border = {"first":first_frame, "last":last_frame}
            tracking_groups = trackingdf.groupby('N')
            #game_mask = skeleton.index.get_level_values(0) == game_id
            flips = getFlips(game_id)
        except Exception as e:
            print(f"Error processing game {game_id}, {traceback.format_exc()}")
        #with tqdm(desc="Processing", leave = False) as pbar:
        for idx, row in tqdm(eventcsv.iterrows(), leave = False):
            action_id = row['EVENT_ID']
            game_id = row['MUID']
            if checkDeadBall(row, eventcsv) == -1:#ensure not a deadball
                continue
            try:
                if ball:
                    all_dfs = getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, border = border, flips = flips, gks = gks, ball = True)
                    speed_dict = all_dfs[0]
                    ball_dict = all_dfs[1]
                    ball_start.loc[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                    ball_start.loc[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                    ball_speeds.loc[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                    ball_speeds.loc[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
                    ball_end.loc[(game_id, action_id), "end_x_a0"] = ball_dict["end_x"]
                    ball_end.loc[(game_id, action_id), "end_y_a0"] = ball_dict["end_y"]
                    
                else:
                    speed_dict = getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, border = border, flips = flips, gks = gks)
            except Exception as e:
                print(f"Error processing game {game_id}, action {action_id}: {traceback.format_exc()}")
                speed_dict = [{"error":None}]
            player_speeds.loc[(game_id, action_id), :] = [None] * len(player_speeds.columns)
            player_speeds.loc[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
        iter += 1
    if ball:
        return player_speeds, ball_speeds, ball_start, ball_end
    return player_speeds
def getBuliFeats(games, output_dir, framesFrom = 10):
    index = pd.MultiIndex.from_tuples([], names=['game_id', 'action_id'])
    success = pd.DataFrame(columns = ["success"], index = index)
    scores = pd.DataFrame(columns = ["scores"], index = index)
    concedes = pd.DataFrame(columns = ["concedes"], index = index)
    scores_xg = pd.DataFrame(columns = ["scores_xg"], index = index)
    concedes_xg = pd.DataFrame(columns = ["concedes_xg"], index = index)
    for game_id in games:
        eventcsv = load_xml.load_csv_event(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
        is_kickoff = (eventcsv['SUBTYPE'] == 'Kickoff') | (eventcsv['SUBTYPE'] == 'FinalWhistle')
        kickoffFrames = eventcsv[is_kickoff].index
        for idx, row in eventcsv.iterrows():
            action_id = row['EVENT_ID']
            if checkDeadBall(row, eventcsv) == -1:
                continue
            feats = getFeatsPlay(idx, row, kickoffFrames, framesFrom)
            feats['success'] = row['EVALUATION'] in ['successfullyComplete', 'successful']
            success.loc[(game_id, action_id), "success"] = feats['success']
            scores.loc[(game_id, action_id), "scores"] = feats['scores']
            concedes.loc[(game_id, action_id), "concedes"] = feats['concedes']
            scores_xg.loc[(game_id, action_id), "scores_xg"] = feats['scores_xg']
            concedes_xg.loc[(game_id, action_id), "concedes_xg"] = feats['concedes_xg']
    success.to_parquet(f"{output_dir}/y_success.parquet")
    scores.to_parquet(f"{output_dir}/y_scores.parquet")
    concedes.to_parquet(f"{output_dir}/y_concedes.parquet")
    scores_xg.to_parquet(f"{output_dir}/y_scores_xg.parquet")
    concedes_xg.to_parquet(f"{output_dir}/y_concedes_xg.parquet")
    return 1
def getFeatsPlay(idx, row, kickoffFrames, framesFrom):
    shots = ['BlockedShot', 'SavedShot', 'SuccessfulShot', 'ShotWide']
    featuresOutput = {"scores": False, "concedes": False, "scores_xg":0, "concedes_xg":0}
    greater_values = [kickoffFrame for kickoffFrame in kickoffFrames if kickoffFrame > idx]
    closest_end = min(greater_values) if greater_values else None
    team = row['CUID1']
    nextidx = min(idx + framesFrom + 1, closest_end)
    nextFrames = eventcsv.loc[idx + 1:nextidx]
    shots = nextFrames[nextFrames['SUBTYPE'].isin(shots)]
    if len(shots) == 0:
        return featuresOutput
    shots['xG'] = shots['xG'].str.replace(",", ".").astype(float)
    offensiveShots = shots[shots['CUID1'] == team]
    featuresOutput["scores_xg"] = 1 - np.prod(1 - offensiveShots['xG'])
    featuresOutput['scores'] = 'SuccessfulShot' in offensiveShots['SUBTYPE']
    defensiveShots = shots[shots['CUID1'] != team]
    featuresOutput["concedes_xg"] = 1 - np.prod(1 - defensiveShots['xG'])
    featuresOutput['concedes'] = 'SuccessfulShot' in defensiveShots['SUBTYPE']
    return featuresOutput


#HAWKEYE FUNCTIONS
def getGksTM(game_id, teams = False):
        lineups = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/lineups/{game_id}.json"
        lineup_df = pd.read_json(lineups, convert_dates = False)
        team_1 = lineup_df['team_id'].loc[0]
        team_2 = lineup_df['team_id'].loc[1]
        team_1_dict = lineup_df['lineup'].loc[0]
        team_2_dict = lineup_df['lineup'].loc[1]
        #print(lineup_df)
        team_1_lineup = [player_dict['player_id'] for player_dict in team_1_dict]
        team_2_lineup = [player_dict['player_id'] for player_dict in team_2_dict]
        team_map = {team_1 : team_1_lineup, team_2 : team_2_lineup}#building a map of team id to player ids
        player_to_team = {player_id: team_id for team_id, players in team_map.items() for player_id in players} #mapping players to teams
        pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_1_dict if len(player['positions']) > 0}
        team_2_pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_2_dict if len(player['positions']) > 0}
        pos_dict.update(team_2_pos_dict)
        goalkeepers = [key for (key,value) in pos_dict.items() if value == "Goalkeeper"]
        if teams:
            return player_to_team, goalkeepers, [team_1, team_2]
        return player_to_team, goalkeepers#gets set of goalkeepers too
from unxpass.converters.conversions import convert_Hawkeye
from timeit import default_timer as timer

def he_ball_speed(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences):
    startx = 0
    starty = 0
    speedx = 0
    speedy = 0
    dummy_set = {"start_x":startx, 
        "start_y": starty,
        "speed_x": speedx,
        "speed_y": speedy,
        "end_x": 0,#don't need end locations since we aren't training, just evaluating
        "end_y":0}
    if sb_action_id in sequences["id"].values:#sequence of interest
        sequence_df = sequences[sequences["id"] == sb_action_id].iloc[0] 
        period = sequence_df["period"]
    else:
        return dummy_set
    time = sequence_df["BallReceipt"]
    if pd.isna(time):
        return dummy_set#ballreceipt can be empty
    time = time + ((int(frame_idx) - 1) * .04)
    start_time = time - (frame_back * .04)#need to account for time before and after half
    end_time = time + (frame_forward * .04)
    times = trackingdf[(trackingdf["elapsed"] >= start_time) & (trackingdf["elapsed"] <= end_time) & (trackingdf['period'] == period)]
    times = times.sort_values(by = ["elapsed"])
    middle = len(times) // 2
    event_time = times[times['elapsed'] == min(times['elapsed'].unique(), key=lambda x:abs(x-time))].iloc[0]['position'].strip("[]").split(", ")
    #print(event_time == times.iloc[middle]['position'].strip("[]").split(", "))
    #event_time = times.iloc[middle]['position']
    #in the "[a,b]" form in string form, need to manually convert - probably a library that does this but whatevs

    #end_time = 
    #may have overlapping values due to added time
    pre_ball_pos = times.iloc[0]['position']
    post_ball_pos = times.iloc[-1]['position']
    event_x = float(event_time[0]) + 105/2
    event_y = 68 - (float(event_time[1]) + 68/2)
    time_elapsed = .04 * (frame_back + frame_forward)
    pre_ball_pos = pre_ball_pos.strip("[]").split(", ")
    post_ball_pos = post_ball_pos.strip("[]").split(", ")
    speedx = (float(post_ball_pos[0]) - float(pre_ball_pos[0])) / time_elapsed
    speedy = (float(post_ball_pos[1]) - float(pre_ball_pos[1])) / time_elapsed
    return {"start_x":event_x, 
        "start_y": event_y,
        "speed_x": speedx,
        "speed_y": speedy,
        "end_x": 0,
        "end_y":0}

def he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball = False, balldf = None):
    output = []
    if ball:
        ball_dict = he_ball_speed(sb_action_id, frame_idx, frame_back, frame_forward, game, balldf, sequences)
    else:
        ball_dict = None
    dummy_set =  {
            "teammate": False,
            "x":0,
            "y":0,
            "player": None,
            "actor": False,
            "goalkeeper":False,
            "x_velo": 0,
            "y_velo": 0
        }
    if sb_action_id in sequences["id"].values:#sequence of interest
        sequence_df = sequences[sequences["id"] == sb_action_id].iloc[0] 
    else:
        if ball:
            return [dummy_set],  ball_dict
        return [dummy_set]#just get first value, dummy data to keep compiler happy
    time = sequence_df["BallReceipt"]
    if pd.isna(time):
        if ball:
            return [dummy_set],  ball_dict
        return [dummy_set]#ballreceipt can be empty
    team = sequence_df["possession_team_id"]
    period = sequence_df['period']
    time = time + ((int(frame_idx) - 1) * .04)
    start_time = time - (frame_back * .04)#need to account for time before and after half
    end_time = time + (frame_forward * .04)
    times = trackingdf.loc[(trackingdf["elapsed"] >= start_time) & (trackingdf["elapsed"] <= end_time) & (trackingdf['period'] == period), "elapsed"].unique()
    times.sort()
    #may have overlapping values due to added time
    start_time = times[0]
    end_time = times[-1]
    #check edge case of x frames back causes to go back to prev period or next period
    #print(start['period'].iloc[0]
    middle = int((frame_back + frame_forward)/2)
    event_time = times[middle]
    actor = sequence_df["player_id"]
    current_tracking = clean_he_frame_df(trackingdf[trackingdf["elapsed"] == event_time], team)
    start_tracking = clean_he_frame_df(trackingdf[trackingdf["elapsed"] == start_time], team)
    end_tracking = clean_he_frame_df(trackingdf[trackingdf["elapsed"] == end_time], team)
    time_elapsed = (frame_back + frame_forward) * .04
    #problem: there are players in start_tracking that are not in end_tracking
    for player in start_tracking["statsbombid"]:
        isActor = actor == player
        goalkeeper = player in gkslist
        start_x = start_tracking[start_tracking["statsbombid"] == player]["x"].values[0]
        start_y = start_tracking[start_tracking["statsbombid"] == player]["y"].values[0]
        end_x = end_tracking[end_tracking["statsbombid"] == player]["x"].values[0]
        end_y = end_tracking[end_tracking["statsbombid"] == player]["y"].values[0]
        x_velo = (end_x - start_x) / time_elapsed
        y_velo = (end_y - start_y) / time_elapsed
        x_loc = current_tracking[current_tracking["statsbombid"] == player]["x"].values[0]
        y_loc = current_tracking[current_tracking["statsbombid"] == player]["y"].values[0]
        location = [end_x, end_y]
        isTeammate = end_tracking[end_tracking["statsbombid"] == player]["isTeammate"].iloc[0]
        #print(isActor)
        speed_dict = {
            "teammate": isTeammate,
            "goalkeeper": goalkeeper,
            "x": x_loc,
            "y": y_loc,
            "actor": isActor,
            "player": player,
            "x_velo": x_velo,
            "y_velo": y_velo
        }
        output.append(speed_dict)

    return getFlip(output, ball_dict)

def getFlip(freezeframe, secondary_frame = None):
    for player in freezeframe:
        if player['teammate'] and player['goalkeeper']:
            attack_gk_x = player['x']
        if not player['teammate'] and player['goalkeeper']:
            defend_gk_x = player['x']
    if attack_gk_x > defend_gk_x: #attacking gk is on wrong side of field, need to flip
        for player in freezeframe:
            player['x'] = 105 - player['x']
            player['x_velo'] = -1 * player['x_velo']
        if secondary_frame:
            secondary_frame["end_x"] = 105 - secondary_frame['end_x']
            secondary_frame["start_x"] = 105 - secondary_frame['start_x']
            secondary_frame["speed_x"] = -1 * secondary_frame["speed_x"]
    if secondary_frame is not None:
        return freezeframe, secondary_frame
    return freezeframe

def convert_Hawkeye(coords):
    """
    Convert hawkeye coords to statsbomb coords
    1. Need to convert from meters to yards - done
    2. Flip and resize such that HawkEye coord system goes to statsbomb coord. system
    3. Flip x axis if needed - dependending on GK location - done
    """
    x, y = coords
    x = float(x)
    y = float(y)
    x = (x + 105/2)
    y = 68 - (y + 34)
    #convert to statsbomb coord system here...
    return [x, y]

def clean_he_frame_df(df, team):
    df = df.copy()
    df.loc[:,"isTeammate"] = df["team"] == int(team)
    needFlip = float(df[(df['isGk']) & (df['isTeammate'])].iloc[0]['position'].strip("[]").split(", ")[0]) > 0
    #df.loc[:,'needsFlip'] = needFlip
    df.loc[:,'position'] = df.apply(lambda row: convert_Hawkeye(row['position'].strip("[]").split(", ")), axis=1)
    df.loc[:,'x'] = df.loc[:,'position'].apply(lambda x: x[0])
    df.loc[:,'y'] = df.loc[:,'position'].apply(lambda x: x[1])
    return df

#Game specific
def getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, db, framesback, framesforward, sequences, ball = False):
    #game_file = "2032219_Portugal_Switzerland.csv"
    game = game_file.split(".")[0]
    
    tracking_path = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/tracking_csvs/{game_file}"
    
    tracking = pd.read_csv(tracking_path)#.sort_values(by = ["elapsed"])
    tracking['statsbombid'] = tracking['uefaId'].astype(int).map(uefa_map)
#3835338, action 7218
    statsbomb_id = hawkeye_to_sb[game]

    game_mask = skeleton.get_level_values(0) == str(statsbomb_id)
    game_skeleton = skeleton[game_mask]
    team_dict, gks = getGksTM(statsbomb_id)
    tracking['team'] = tracking['statsbombid'].map(team_dict)
    tracking['isGk'] = tracking['role'] == "Goalkeeper"
    speed_df = pd.DataFrame(index = game_skeleton)
    action_df = db.actions(game_id = int(statsbomb_id))
    action_map = pd.Series(action_df['original_event_id'].values, index=action_df.index).to_dict()
    player_speeds = pd.DataFrame(index = game_skeleton)
    if ball:
        ball_tracking_path = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/tracking_ball_csvs/{game_file}"
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
        
        #start = timer()    
        try:
            if ball:
                speed_dict, ball_dict = he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball, ball_df)
                
                ball_starts.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                ball_starts.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
            else:
                speed_dict = he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball)
            player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict

        except Exception as e:
            res = dict((v,k) for k,v in hawkeye_to_sb.items())
            print(f"Error processing game {game_id}, {res[int(game_id)]}, action {action_id}: {traceback.format_exc()}")
            speed_dict = {}
    if ball:
        return player_speeds, ball_starts, ball_speeds
    return player_speeds

def getHeSpeed(tracking_folder, skeleton_path, dbpath, framesback, framesforward, ball = False):
    player_speeds = []
    ball_starts = []
    ball_speeds = []
    sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
    timeelapsed = {
    1:0,
    2:45 * 60,
    3: 90 * 60,
    4: 105 * 60
    }
    sequences["BallReceipt"] = sequences["period"].map(timeelapsed) + sequences["BallReceipt"]#minute adjustment
    with open("/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    skeleton = pd.read_parquet(skeleton_path).index
    hawkeye_db = SQLiteDatabase(dbpath)
    framesback = 5
    framesforward = 5
    alltracking = [file for file in os.listdir(tracking_folder) if file.endswith(".csv")]
    uefa_map = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/player_ids_matched.csv")
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
#sequences

def generate_Hawkeye_From_Features(output_dir, frame_forward = 5, frame_back = 5, ball = False, frame_idxs = [1]):
    uefa_map = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/player_ids_matched.csv")
    uefa_map = pd.Series(uefa_map["sb_player_id"].values,index=uefa_map["uefa_player_id"]).to_dict()
    frame_forward, frame_back = 5,5
    
    with open("/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    sb_to_hawkeye = dict((v,k) for k,v in hawkeye_to_sb.items())
    minute_adjustment = {
    1: 0,
    2: 45 * 60,
    3: 90 * 60,
    4: 105 * 60
    }
    sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
    sequences = sequences.rename(columns = {"Half":"period"})
    sequences["hawkeye_game_id"] = sequences["match_id"].map(sb_to_hawkeye)
    sequences["BallReceipt"] = sequences["period"].map(minute_adjustment) + sequences["BallReceipt"]
    sequences["Start"] = sequences["period"].map(minute_adjustment) + sequences["Start"]
    frame_path = f"{output_dir}/x_freeze_frame_360.parquet"
    frame_dfs = []
    if ball:
        ball_start_output = f"{output_dir}/x_startlocation.parquet"
        ball_speed_output = f"{output_dir}/x_speed.parquet"
        ball_end_output = f"{output_dir}/x_endlocation.parquet"
        ball_start_dfs = []
        ball_speed_dfs = []
        ball_end_dfs = []
    
    for game in tqdm(sequences['hawkeye_game_id'].unique()[:1]):
        for frame_idx in tqdm(frame_idxs):
            if ball:
                player_speeds, ball_starts, ball_speeds, ball_end = hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idx, frame_back, frame_forward, ball)
                ball_start_dfs.append(ball_starts)
                ball_speed_dfs.append(ball_speeds)
                ball_end_dfs.append(ball_end)
            else:
                player_speeds = hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idx, frame_back, frame_forward, ball)
            frame_dfs.append(player_speeds)
    if ball:
        combined_ball_start = pd.concat(ball_start_dfs)
        combined_ball_speed = pd.concat(ball_speed_dfs)
        combined_ball_end = pd.concat(ball_end_dfs)

        combined_ball_start.to_parquet(ball_start_output)
        combined_ball_speed.to_parquet(ball_speed_output)
        combined_ball_end.to_parquet(ball_end_output)
    combined_frame_dfs = pd.concat(frame_dfs)
    combined_frame_dfs.to_parquet(frame_path)

def hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idx = 1, frame_back = 5, frame_forward = 5, ball = False):
    #need to make adjustment for passes
    sequence_games = sequences[sequences['hawkeye_game_id'] == game].copy()
    sequence_games.loc[:, 'index'] = sequence_games.loc[:, 'index'].astype(str) + f"-{int(frame_idx) - 1}"
    multiindex = pd.MultiIndex.from_frame(sequence_games[['match_id', 'index']])
    player_speeds = pd.DataFrame(index = multiindex)
    player_speeds["freeze_frame_360_a0"] = np.nan
    player_speeds["freeze_frame_360_a0"] = player_speeds["freeze_frame_360_a0"].astype(object)
    if ball:
        ball_starts = pd.DataFrame(index = multiindex)
        ball_speeds = pd.DataFrame(index = multiindex)
        ball_end = pd.DataFrame(index = multiindex)
    tracking_path = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/tracking_csvs/{game}.csv"
    statsbombid = hawkeye_to_sb[game]
    team_dict, gkslist, teams = getGksTM(statsbombid, True)
    trackingdf = pd.read_csv(tracking_path)
    trackingdf['statsbombid'] = trackingdf['uefaId'].astype(int).map(uefa_map)
    trackingdf['team'] = trackingdf['statsbombid'].map(team_dict)
    #BAND-AID Solution - idk what else to do tbh
    if teams[0] not in trackingdf['team'].values and teams[1] not in trackingdf['team'].values:
        raise Exception("No Team Found")
    elif teams[0] not in trackingdf['team'].values:
        trackingdf['team'] = trackingdf['team'].fillna(teams[0])
    elif teams[1] not in trackingdf['team'].values:
        trackingdf['team'] = trackingdf['team'].fillna(teams[1])
    trackingdf['isGk'] = trackingdf['role'] == "Goalkeeper" 
    for idx, row in tqdm(sequence_games.iterrows(), leave = False):
        sb_action_id = row['id']
        action_id = row['index']
        game_id = row['match_id']
        if ball:
            ball_tracking_path = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/tracking_ball_csvs/{game}.csv"
            ball_df = pd.read_csv(ball_tracking_path)
            speed_dict, ball_dict = he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball, ball_df)
            #
            ball_starts.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
            ball_starts.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
            ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
            ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
            ball_end.at[(game_id, action_id), "end_x_a0"] = 60
            ball_end.at[(game_id, action_id), "end_y_a0"] = 40#dummy to center
        else:
            speed_dict = he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball)
        player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
    if ball:
        return player_speeds, ball_starts, ball_speeds, ball_end
    return player_speeds
def getDummyLabels(output_dir, dummy_idxs):
    concedes_xg = f"{output_dir}/y_concedes_xg.parquet"
    concedes = f"{output_dir}/y_concedes.parquet"
    scores_xg = f"{output_dir}/y_scores_xg.parquet"
    scores = f"{output_dir}/y_scores.parquet"
    success = f"{output_dir}/y_success.parquet"
    c_xg = pd.DataFrame(index = dummy_idxs)
    c = pd.DataFrame(index = dummy_idxs)
    s_xg = pd.DataFrame(index = dummy_idxs)
    s = pd.DataFrame(index = dummy_idxs)
    suc = pd.DataFrame(index = dummy_idxs)
    for idx in dummy_idxs:
        c_xg.at[idx, "concedes_xg"] = 0
        c.at[idx, "concedes"] = False
        s_xg.at[idx, "scores_xg"] = 0
        s.at[idx, "scores"] = False
        suc.at[idx, "success"] = True
    c_xg.to_parquet(concedes_xg)
    c.to_parquet(concedes)
    s_xg.to_parquet(scores_xg)
    s.to_parquet(scores)
    suc.to_parquet(success)



def main(buli, hawkeye, hawkeye_raw, ball):
    #RUN BULI
    if buli:
        print("Adding Bundesliga Speeds...")
        games = [game_id.split("_")[-1].split(".")[0] for game_id in os.listdir("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/")]
        games = games[:1]
        #skeleton = pd.read_parquet("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/all_features/x_endlocation_og.parquet")
        #dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
        feat_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_test"
        #db = SQLiteDatabase(dbpath)
        
        if ball:
            #player_speeds, ball_speeds, ball_starts, ball_end
            dfs = addAllSpeedBuli(games, ball)
            buli_speed = dfs[0]
            dfs[1].to_parquet(f"{feat_path}/x_speed.parquet")
            dfs[2].to_parquet(f"{feat_path}/x_startlocation.parquet")
            dfs[3].to_parquet(f"{feat_path}/x_endlocation.parquet")
        else:
            buli_speed = addAllSpeedBuli(games, ball)
        buli_speed.to_parquet(f"{feat_path}/x_freeze_frame_360.parquet")
        getBuliFeats(games)
        #print(buli_speed)
        #buli_speed.to_csv("testing_speed_frame.csv")
        

    if hawkeye:
        #generate from pregenerated statsbomb features
        dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
        tracking_folder = "/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkeye/tracking_csvs"
        skeleton_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_endlocation.parquet"
        output_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_frame_360.parquet"
        #player_speed_df.loc[skeleton], ball_start_df.loc[skeleton], ball_speeds_df.loc[skeleton]
        if ball:
            all_dfs = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
            ball_start_output = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_startlocation_3.parquet"
            ball_speed_output = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_3.parquet"
            speeddf = all_dfs[0]
            all_dfs[1].to_parquet(ball_start_output)
            all_dfs[2].to_parquet(ball_speed_output)
        #print(speeddf)
        else:
            speeddf = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
        speeddf.to_parquet(output_path)
    if hawkeye_raw:
        #generate completely from hawkeye data - recommend this
        output_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Hawkeye_Features/Hawkeye_Features_Updated_wSecond_test"
        sequence_games = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
        generate_Hawkeye_From_Features(output_dir, ball = ball, frame_idxs = range(1,27))
        dummy_idxs = pd.read_parquet(f"{output_dir}/x_startlocation.parquet").index
        getDummyLabels(output_dir, dummy_idxs)
#main(buli, hawkeye, hawkeye_raw, ball)
if __name__ == '__main__':  main(True, False, False, True)
