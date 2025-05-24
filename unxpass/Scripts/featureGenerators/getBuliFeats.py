#Generates features from Bundesliga Data
from unxpass import load_xml
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from unxpass.databases import SQLiteDatabase
import traceback
from time import process_time
def getFlip(freezeframe, secondary_frame = None):
    """ 
    Determines if a flip is needed by gk location
    """
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
def frametodict(group):
    """
    Converts a group of tracking data into a dictionary with player IDs as keys and their translated positions.
    """
    # Exclude the BALL rows and work on a copy to avoid SettingWithCopyWarning
    if len(group) == 0:
        raise ValueError("Empty group provided for conversion.")
    tracking = group.copy()

    # Ensure numeric conversion for X and Y (if not already floats)
    tracking['X'] = tracking['X'].astype(float)
    tracking['Y'] = tracking['Y'].astype(float)
    
    # Compute translated coordinates using vectorized operations
    tracking['X_translated'] = (tracking['X'] + 105/2)
    tracking['Y_translated'] = 68 - (tracking['Y'] + 34)#features are already in meters, no need to convert
    
    
    locs = {
        row.PersonId: {"X": row.X_translated, "Y": row.Y_translated, "Team": row.TeamId}
        for row in tracking.itertuples(index=False)
    }
    
    return locs


def getAngleFrames(sample_pass, allAngles):
    """
    Determining frame of interception, where either
    1. Change of angle is above 20
    2. Out of bounds
    """
    start_frame = sample_pass['FRAME_NUMBER']
    event_id = sample_pass['EVENT_ID']
    outOfBounds = (abs(allAngles['X'].astype(float)) > 52.5) | (abs(allAngles['Y'].astype(float)) > 34)
    angleStartFrame = allAngles[((allAngles['angle_change'] > 20) & (allAngles['N'] > start_frame + 10))  | ((outOfBounds) & (allAngles['N'] > start_frame + 5)) ]
    if len(angleStartFrame) == 0:
        return start_frame + (2 / .04) #get 2 seconds after pass
    angleStartFrame = angleStartFrame.loc[angleStartFrame['N'].idxmin()]['N']
    return angleStartFrame

def getAngleChangeFromStart(tracking, pass_start_map, fb = 10):
    """
    Compute angle change between ball direction at pass start and at each frame.

    - tracking: DataFrame with ball tracking data (must contain 'N', 'X', 'Y', 'GameSection', 'TeamId')
    - pass_start_map: dict or Series mapping frame N to pass start frame N0

    """
    ball_data_df = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball_data_df['N'] = ball_data_df['N'].astype(float)
    ball_data_df['X'] = ball_data_df['X'].astype(float)
    ball_data_df['Y'] = ball_data_df['Y'].astype(float)
    ball_data_df = ball_data_df.sort_values(by='N')


    ball_data_df['dx'] = ball_data_df['X'] - ball_data_df.groupby('GameSection')['X'].shift(10)
    ball_data_df['dy'] = ball_data_df['Y'] - ball_data_df.groupby('GameSection')['Y'].shift(10)
    ball_data_df['magnitude'] = np.sqrt(ball_data_df['dx']**2 + ball_data_df['dy']**2)


    ball_data_df['pass_start_N'] = ball_data_df['N'].map(pass_start_map)
    
    ball = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball[['N', 'X', 'Y']] = ball[['N', 'X', 'Y']].astype(float)
    ball = ball.sort_values('N')
    ball = ball.drop_duplicates(subset='N')  # Ensure no duplicate frames
    ball_indexed = ball.set_index('N')


    start_pos = ball_indexed.reindex(ball_data_df['pass_start_N'])
    end_pos = ball_indexed.reindex(ball_data_df['pass_start_N'] + fb)


    start_dxdy = end_pos[['X', 'Y']].values - start_pos[['X', 'Y']].values

    ball_data_df[['start_dx', 'start_dy']] = start_dxdy
    ball_data_df['start_magnitude'] = np.hypot(ball_data_df['start_dx'], ball_data_df['start_dy'])
    
    ball_data_df['dot_product'] = (ball_data_df['dx'] * ball_data_df['start_dx'] +
                                   ball_data_df['dy'] * ball_data_df['start_dy'])

    cosine_values = ball_data_df['dot_product'] / (ball_data_df['magnitude'] * ball_data_df['start_magnitude'])
    cosine_values = np.clip(cosine_values, -1, 1)
    ball_data_df['angle_change'] = np.degrees(np.arccos(cosine_values))

    # Clean up infinities and NaNs
    ball_data_df['angle_change'].replace([np.inf, -np.inf], np.nan, inplace=True)
    ball_data_df['angle_change'].fillna(0, inplace=True)

    return ball_data_df[['X', 'Y', 'angle_change', 'N','pass_start_N', 'GameSection', 'start_dx', 'start_dy', 'start_magnitude']]
import regex as re
def getClosestFrame(row, tracking, angle):
    """
    Determines frame of reception / interception
    """
    if row["SUBTYPE"] != "Pass" and row['SUBTYPE'] != "Cross":
        return row["RECFRM"]#if not pass / cross - essentiallly None
    if not pd.isna(row["RECFRM"]):
        return row["RECFRM"]#if already has receipt frame from event data
    #if we don't know recipient or the evaluation is false - use 20 degree method    
    if pd.isna(row["PUID2"]) or row["EVALUATION"] == "unsuccessful":
        angleStartFrame = getAngleFrames(row, angle)
        if abs(angleStartFrame - row["FRAME_NUMBER"]) < (2 / .04):
            return angleStartFrame#check if within 2 seconds
        if pd.isna(row["PUID2"]):
            return row["FRAME_NUMBER"] + (2/.04) #if no recipient or angle frame, get frame 2 seconds from

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

    n_values = df1['N'].values

    indices = np.searchsorted(frame_vals, n_values, side='right') - 1
    indices[indices < 0] = 0
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
    eventdf["NEXT_FRAME"] = eventdf["NEXT_FRAME"].fillna(method='bfill')
    #if the difference between two events is over 300, then default to the frame(practically only occurs during halftime and fulltime whistles, or with missing events)
    eventdf.loc[abs(eventdf["NEXT_FRAME"] - eventdf["FRAME_NUMBER"]) > 300, "NEXT_FRAME"] = eventdf.loc[abs(eventdf["NEXT_FRAME"] - eventdf["FRAME_NUMBER"]) > 300, "FRAME_NUMBER"]
    tracking["N"] = tracking["N"].astype(float)
    tracking["X"] = tracking["X"].astype(float)
    tracking["Y"] = tracking["Y"].astype(float)
    #gets closest frame
    eventdf["RECFRM"] = eventdf.apply(lambda row: getClosestFrame(row, tracking, angles), axis = 1)
    return eventdf
def getOutOfBounds(row, tracking):
    """
    Determine if out of bounds
    """
    outOfBounds = (abs(tracking['X']) > 52.5) | (abs(tracking['Y']) > 34)
    start_frame = row["FRAME_NUMBER"]
    end_frame = row['RECFRM'] + 300
    ball_data = tracking[(tracking["N"] >= start_frame) & (tracking["N"] <= end_frame) & (tracking["TeamId"] == "BALL") & (outOfBounds)]
    if len(ball_data) > 0:
        return min(ball_data["N"])
    else:
        return row['RECFRM']
def getSpeedBuli(action_id, tracking_groups, eventcsv, border,  gks, framesback = 5, framesforward = 5, ball = False, checkBlocked = False):
    """
    Generates freezeframe and player velocities
    """
    timediff = 0.04 * (framesback + framesforward)
    buli_id = action_id
    event = eventcsv[eventcsv['EVENT_ID'] == buli_id].iloc[0]
    successful = event['EVALUATION'] in ['successful', 'successfullyComplete']
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
    if checkBlocked:#checkBlocked ensures that there are atleast 10 frames between the start and end of a pass
        nextTen = event_frame + 10
        blocked = endFrame < nextTen
        endFrame = nextTen
    event_frame_str = min(event_frame + framesforward, period_end)
    prior_frame_str = max(event_frame - framesback, period_start)
    ballPrior =  max(event_frame - 2 * framesback, period_start)#goes framesback * 2(10 in this case) frames back
    end_frame_str = min(endFrame, period_end)
    #edge case of half time occuring
    event_pos = tracking_groups.get_group(event_frame_str)
    prior_pos = tracking_groups.get_group(prior_frame_str)
    ballPrior = tracking_groups.get_group(ballPrior)

    event_pos = frametodict(event_pos)
    prior_pos = frametodict(prior_pos)
    current_pos = frametodict(current_pos)
    ballPrior = frametodict(ballPrior)
    
    ball_ff = None
    if ball:
        end_pos = tracking_groups.get_group(end_frame_str)
        end_pos = frametodict(end_pos)["DFL-OBJ-0000XT"]
        event_pos_ball = event_pos["DFL-OBJ-0000XT"]
        prior_pos_ball = ballPrior["DFL-OBJ-0000XT"]
        current_pos_ball = current_pos["DFL-OBJ-0000XT"]
        x_velo = (event_pos_ball["X"] - prior_pos_ball["X"]) / timediff
        y_velo = (event_pos_ball["Y"] - prior_pos_ball["Y"]) / timediff
        if checkBlocked and blocked and not successful:#if the ball is blocked and we are looking to remove blocked passes
            end_pos['X'] = -100000#some arbitrary stupid value to check and drop later
            end_pos['Y'] = -100000
        ball_ff = {
            "start_x": current_pos_ball["X"],
            "start_y": current_pos_ball["Y"],
            "speed_x": x_velo,
            "speed_y": y_velo,#since we are dropping speed, this does not matter
            "end_x": end_pos["X"],
            "end_y":end_pos["Y"]
        }
    speed_output = []
    for player, pos in event_pos.items():
        if player != "DFL-OBJ-0000XT":
            isTeammate = event_pos[player]["Team"] == team
            isActor = actor == player
            #print(isActor)
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
                "x": current_pos[player]["X"], 
                "y": current_pos[player]["Y"],
                "goalkeeper":isGoalKeeper,
                "recipient": isRecipient
            }
            speed_output.append(player_dict)
    return getFlip(speed_output, ball_ff)#ensure that properly flipped, double checking
def checkDeadBall(row, eventcsv):
    """
    Ensure pass is not from a deadball situation
    """
    deadBalls = ['Kickoff', 'ThrowIn', 'FreeKick','GoalKick']
    if row['SUBTYPE'] not in ['Pass', 'Cross']:#ensure is a legitimate pass
        return -1
    if (eventcsv['FRAME_NUMBER'] == row['FRAME_NUMBER']).sum() > 1:
        framecsv = eventcsv[eventcsv['FRAME_NUMBER'] == row['FRAME_NUMBER']]
        if framecsv['SUBTYPE'].isin(deadBalls).any():
            return -1#ensure also not a deadball situation
    return 1
def addAllSpeedBuli(games, ball = False, checkBlocked = False):
    """
    Generates speeds and features for all bundesliga data
    Games is a list of game ids
    ball is a boolean to determine if ball speeds and features are included
    checkBlocked is a boolean to determine whether to check for blocked passes if using the 10 frame rule
    """
    #
    indices = []
    print("Generating Indices")#pregenerates indices for all games, used to speed up process of appending to dataframe
    for game_id in tqdm(games):
        eventcsv = load_xml.load_csv_event(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
        for idx, row in tqdm(eventcsv.iterrows(), leave = False):
            action_id = row['EVENT_ID']
            if checkDeadBall(row, eventcsv) == -1:#ensure not a deadball situation
                continue
            indices.append((game_id, action_id))
    index = pd.MultiIndex.from_tuples(indices, names=['game_id', 'action_id'])
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
            eventcsv = load_xml.load_csv_event(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
            trackingdf = pd.read_csv(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/tracking_csv/{game_id}.csv")
            lineups = load_xml.load_players(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/match_information/{game_id}.xml", False)
            first_frame = {}
            last_frame = {}
            eventcsv = getReceipts(trackingdf, eventcsv)
            gks =  lineups[lineups['PlayingPosition'] == "TW"]["PersonId"]
            #none of these games should go to extra time
            period = trackingdf["GameSection"].unique()[0]
            for period in trackingdf["GameSection"].unique():#probably can be optimized by a groupBy
                first_frame[period] = trackingdf[trackingdf["GameSection"] == period]["N"].astype(int).min()
                last_frame[period] = trackingdf[trackingdf["GameSection"] == period]["N"].astype(int).max()
            border = {"first":first_frame, "last":last_frame}
            tracking_groups = trackingdf.groupby('N')
            #game_mask = skeleton.index.get_level_values(0) == game_id
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
                    all_dfs = getSpeedBuli( action_id, tracking_groups, eventcsv, border = border,gks = gks, ball = True, checkBlocked = checkBlocked)
                    speed_dict = all_dfs[0]
                    ball_dict = all_dfs[1]
                    ball_start.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                    ball_start.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                    ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                    ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
                    ball_end.at[(game_id, action_id), "end_x_a0"] = ball_dict["end_x"]
                    ball_end.at[(game_id, action_id), "end_y_a0"] = ball_dict["end_y"]
                    
                else:
                    speed_dict = getSpeedBuli(game_id, action_id, tracking_groups, eventcsv, border = border, gks = gks, checkBlocked = checkBlocked)
            except Exception as e:
                print(f"Error processing game {game_id}, action {action_id}: {traceback.format_exc()}")
                speed_dict = [{"error":None}]
            player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
        iter += 1
    if ball:
        return player_speeds, ball_speeds, ball_start, ball_end
    return player_speeds
def getBuliLabels(games, output_dir, framesFrom = 10, xgType = "xml"):
    """
    Generates labels for training from bundesliga data
    framesFrom describes how far into the future to look for shots
    xgType - since the xG from the xml and csv files are different, determine which one to use
    """
    indices = []
    print("Generating Indices")
    for game_id in tqdm(games):
        eventcsv = load_xml.load_csv_event(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
        for idx, row in tqdm(eventcsv.iterrows(), leave = False):
            action_id = row['EVENT_ID']
            if checkDeadBall(row, eventcsv) == -1:#ensure not a deadball
                continue
            indices.append((game_id, action_id))
    index = pd.MultiIndex.from_tuples(indices, names=['game_id', 'action_id'])
    success = pd.DataFrame(columns = ["success"], index = index)
    scores = pd.DataFrame(columns = ["scores"], index = index)
    concedes = pd.DataFrame(columns = ["concedes"], index = index)
    scores_xg = pd.DataFrame(columns = ["scores_xg"], index = index)
    concedes_xg = pd.DataFrame(columns = ["concedes_xg"], index = index)
    nonActions = ['Offside', 'Substitution', 'Caution', 'FairPlay', 'Nutmeg', 'PossessionLossBeforeGoal', 'BallDeflection',
    'VideoAssistantAction']
    for game_id in tqdm(games):
        eventcsv = load_xml.load_csv_event(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv")
        xmlEvent = load_xml.load_event(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/event_data_all/{game_id}.xml")
        xmlEvent = xmlEvent[['EventId', 'xG']].rename(columns = {'xG':"xml_xG"})
        
        eventcsv = eventcsv[~eventcsv['SUBTYPE'].isin(nonActions)]#remove all nonactions
        xmlEvent['EventId'] = xmlEvent['EventId'].astype(int)
        eventcsv = pd.merge(eventcsv, xmlEvent, left_on = "EVENT_ID", right_on = "EventId")
        half_starts = [10000, 100000, 200000, 250000]
        is_kickoff = (eventcsv['SUBTYPE'] == 'Kickoff')
        kickoffFrames = list(eventcsv[is_kickoff]['FRAME_NUMBER']) + half_starts        
        eventcsv['FRAME_NUMBER'] = eventcsv['FRAME_NUMBER'].astype(float)
        eventcsv = eventcsv.sort_values(by = "FRAME_NUMBER")
        eventcsv = eventcsv.reset_index(drop = True)
        for idx, row in tqdm(eventcsv.iterrows()):
            action_id = row['EVENT_ID']
            if checkDeadBall(row, eventcsv) == -1:
                continue
            feats = getFeatsPlay(idx, row, kickoffFrames, eventcsv, xgType = xgType)
            feats['success'] = row['EVALUATION'] in ['successfullyComplete', 'successful']
            success.at[(game_id, action_id), "success"] = feats['success']
            scores.at[(game_id, action_id), "scores"] = feats['scores']
            concedes.at[(game_id, action_id), "concedes"] = feats['concedes']
            scores_xg.at[(game_id, action_id), "scores_xg"] = feats['scores_xg']
            concedes_xg.at[(game_id, action_id), "concedes_xg"] = feats['concedes_xg']
    success.to_parquet(f"{output_dir}/y_success.parquet")
    scores.to_parquet(f"{output_dir}/y_scores.parquet")
    concedes.to_parquet(f"{output_dir}/y_concedes.parquet")
    scores_xg.to_parquet(f"{output_dir}/y_scores_xg.parquet")
    concedes_xg.to_parquet(f"{output_dir}/y_concedes_xg.parquet")
def getNextNFrames(df, start_idx, closest_end, nextActs = 10):
    """
    Gets the next nextActs actions
    """
    start_frame = df.loc[start_idx, 'FRAME_NUMBER']
    df_after = df.loc[start_idx + 1:]
    next_frames = df_after[df_after['FRAME_NUMBER'] != start_frame]['FRAME_NUMBER'].drop_duplicates().head(nextActs)
    # Filter all rows in df with those 10 FRAME_NUMBERs
    nextEvents = df[df['FRAME_NUMBER'].isin(next_frames)]
    return nextEvents[nextEvents['FRAME_NUMBER'] < closest_end]
def getFeatsPlay(idx, row, kickoffFrames,eventcsv, xgType = "csv"):
    """
    Generates labels for Bundesliga data for each play
    """
    shots = ['ShotWoodWork','OtherShot', 'BlockedShot', 'SavedShot', 'SuccessfulShot', 'ShotWide']
    frame = row['FRAME_NUMBER']
    featuresOutput = {"scores": False, "concedes": False, "scores_xg":0, "concedes_xg":0}
    greater_values = [kickoffFrame for kickoffFrame in kickoffFrames if kickoffFrame > frame]
    closest_end = min(greater_values) if len(greater_values) > 0 else None#get smallest kickoffFrame larger
    team = row['CUID1']
    nextFrames = getNextNFrames(eventcsv, idx, closest_end)
    shots = nextFrames[~pd.isna(nextFrames['xG'])].copy()
    if len(shots) == 0:
        return featuresOutput
    xgCol = "xG" if xgType == "csv" else "xml_xG"
    shots['xG'] = shots['xG'].str.replace(",", ".").astype(float)
    shots['xml_xG'] = shots['xml_xG'].astype(float)
    offensiveShots = shots[shots['CUID1'] == team]
    featuresOutput["scores_xg"] = 1 - np.prod(1 - offensiveShots[xgCol])
    featuresOutput['scores'] = 'SuccessfulShot' in offensiveShots['SUBTYPE'].values
    defensiveShots = shots[shots['CUID1'] != team]
    featuresOutput["concedes_xg"] = 1 - np.prod(1 - defensiveShots[xgCol])
    featuresOutput['concedes'] = 'SuccessfulShot' in defensiveShots['SUBTYPE'].values
    return featuresOutput
def main(ball, checkBlocked):
    #If ball is false - then only freeze frame is created, if not, then all other features are generated
    #checkBlocked - if True, then sets end location as the ball 10 frames fron the start(or -10000 if ball ends before 10 frames)
    print("Getting Bundesliga Features")
    games = [game_id.split(".")[0] for game_id in os.listdir("../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/tracking_csv")]
    games = games[:1]
    feat_path = "../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_ogEndLoc_test"
    if ball:
       dfs = addAllSpeedBuli(games, ball, checkBlocked = checkBlocked)
       buli_speed = dfs[0]
       dfs[1].to_parquet(f"{feat_path}/x_speed.parquet")
       dfs[2].to_parquet(f"{feat_path}/x_startlocation.parquet")
       dfs[3].to_parquet(f"{feat_path}/x_endlocation.parquet")
    else:
       buli_speed = addAllSpeedBuli(games, ball, checkBlocked = checkBlocked)
    buli_speed.to_parquet(f"{feat_path}/x_freeze_frame_360.parquet")
    print("Getting Bundesliga Labels")
    getBuliLabels(games, feat_path, xgType = "xml")
if __name__ == "__main__": main(True, False)