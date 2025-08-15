#Generates features to run model for SkillCorner Data
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from unxpass.Scripts.featureGenerators import getHawkeyeFeats
def getClosestNonNone(frame, bound, tracking, frameObj, isLower = True):
    # For calculating velocity, if frameObj is None or empty, search for closest valid frame.
    # frameObj is the object describing the frame (dict if ball, list if players)

    if isinstance(frameObj, dict):
        colName = "ball_data"
        def is_invalid(obj): return obj['x'] is None
    elif isinstance(frameObj, list):
        colName = "player_data"
        def is_invalid(obj): return len(obj) == 0
    else:
        raise Exception("Invalid DType for frameObj")
    if isLower:
        bound = -1 * bound
    i = 1
    # Forward search
    while is_invalid(frameObj) and i < 10:
        if isLower:#if looking at lower bound
            next_frame = frame + bound - i#keep on going lower
        else:#if looking at upper bound
            next_frame = frame + bound + i #keep on going up
        if next_frame in tracking['frame'].values:
            frameObj = tracking[tracking['frame'] == next_frame].iloc[0][colName]
        i += 1
    i = 1
    # Backward search if still invalid
    while is_invalid(frameObj) and i < abs(bound):
        if isLower:#if looking at lower bound
            prev_frame = frame + bound + i#go higher until frame
        else:#if looking at upper bound
            prev_frame = frame + bound - i #go lower until frame
        if prev_frame in tracking['frame'].values:
            frameObj = tracking[tracking['frame'] == prev_frame].iloc[0][colName]
        i += 1
    return frameObj

    
def getBallData(row, tracking, periodBounds, frameDiff = 5):
    frame = row['frame']
    periodBound = periodBounds[int(row['period'])]
    orientation = 1 if row['attacking_side'] == 'left_to_right' else -1
    lowerBound = min(frameDiff, frame - periodBound[0])#ensure always within half bounds
    upperBound = min(frameDiff, periodBound[1] - frame)
    timeElapsed = 2 * (lowerBound + upperBound) * .1
    ballPrevFive = tracking[tracking['frame'] == frame - lowerBound].iloc[0]['ball_data']
    ballPostFive = tracking[tracking['frame'] == frame + upperBound].iloc[0]['ball_data']
    
    ballData = tracking[tracking['frame'] == frame].iloc[0]['ball_data']
    if ballPrevFive['x'] is None:#this if statement is redundant but kept for code clarity
        ballPrevFive = getClosestNonNone(frame, lowerBound, tracking, ballPrevFive, True)
    if ballPostFive['x'] is None:
        ballPostFive = getClosestNonNone(frame, upperBound, tracking, ballPostFive, False)
    if ballData['x'] is None or ballPrevFive['x'] is None or ballPostFive['x'] is None:#if tracking data is invalid - should be dropped by editFeatures.py
        return pd.Series({"speedx_a02":None, "speed_y_a02":None, 
    "start_x_a0":None, "start_y_a0":None,
    "end_x_a0":None, "end_y_a0":None})
    speed_x = (ballPostFive['x'] - ballPrevFive['x']) / timeElapsed * orientation
    speed_y = (ballPostFive['y'] - ballPrevFive['y']) / timeElapsed
    start_x_a0 = (105 * ((orientation + 1) / 2)) - (ballData['x'] + 52.5)
    start_y_a0 = ballData['y'] + 34
    return pd.Series({"speedx_a02":speed_x, "speed_y_a02":speed_y, 
    "start_x_a0":start_x_a0, "start_y_a0":start_y_a0,
    "end_x_a0":52.5, "end_y_a0":34})#dummy to set to halfLine
def getPlayerFrame(row, tracking, goalkeepers, team_dict,periodBounds, frameDiff = 5):
    #need to add 52.5, 34 and then flip if needed so that always l to r, -1 if l to r, 1 if r to l
    frame = row['frame']
    actor = row['player_id']
    periodBound = periodBounds[int(row['period'])]
    team = row['team_id']
    target = row['player_targeted_id']
    orientation = 1 if row['attacking_side'] == 'left_to_right' else -1
    lowerBound = min(frameDiff, frame - periodBound[0])#ensure always within half bounds
    upperBound = min(frameDiff, periodBound[1] - frame)
    timeElapsed = 2 * (lowerBound + upperBound) * .1
    playerPrevFive = tracking[tracking['frame'] == frame - lowerBound].iloc[0]['player_data']
    playerPostFive = tracking[tracking['frame'] == frame + upperBound].iloc[0]['player_data']
    if len(playerPrevFive) == 0:#this if statement is redundant but kept for code clarity
        playerPrevFive = getClosestNonNone(frame, lowerBound, tracking, playerPrevFive, True)
    if len(playerPostFive) == 0:
        playerPostFive = getClosestNonNone(frame, lowerBound, tracking, playerPostFive, False)
    
    playerData = tracking[tracking['frame'] == frame].iloc[0]['player_data']
    if len(playerData) == 0 or len(playerPostFive) == 0 or len(playerPrevFive) == 0:
        return []#if still cannot find valid frame or passing frame is bad, then just return empty and drop later
    playerFrames = []
    for player in playerPrevFive:
        player_dict = {}
        player_id = player['player_id']
        playerPost = [playerPost for playerPost in playerPostFive if playerPost['player_id'] == player_id][0]
        #ensure no errors if player not avaliable 10 frames later
        player_dict['x_velo'] = ((player['x'] - playerPost['x']) / (timeElapsed)) * orientation
        player_dict['y_velo']  = (player['y'] - playerPost['y']) / (timeElapsed)
        player_dict['x'] = (105 * ((orientation + 1) / 2)) - (player['x'] + 52.5)#map -1 to 0 and 1 to 1
        player_dict['y'] = player['y'] + 34
        player_dict['actor'] = player_id == actor
        player_dict['recipient'] = player_id == target
        player_dict['teammate'] = team_dict[player_id] == team
        player_dict['goalkeeper'] = player_id in goalkeepers
        playerFrames.append(player_dict)
    return playerFrames

def getAllSkillCornerFF(tracking, passes, goalkeepers, team_dict):
    periodBounds = tracking[tracking['player_data'].apply(len) > 0].groupby('period').agg(
        end=('frame', np.max),
        start=('frame', np.min))
    periodDict = {period: [row['start'], row['end']] for period, row in periodBounds.to_dict('index').items()}
    passes['freeze_frame_360_a0'] = passes.progress_apply(lambda row: getPlayerFrame(row, tracking, goalkeepers, team_dict,periodDict, frameDiff = 5), axis = 1)
    passes[["start_x_a0", "start_y_a0", "speed_x_a02", "speed_y_a02","end_x_a0", "end_y_a0"]] = passes.apply(lambda row: getBallData(row, tracking, periodDict, frameDiff = 5), axis = 1)
    passes['game_id'] = passes['match_id']
    passes['action_id'] = passes['event_id']
    passes = passes.set_index(['game_id', 'action_id'])
    freezeFrame = passes['freeze_frame_360_a0']
    startLoc = passes[["start_x_a0", "start_y_a0"]]
    endLoc = passes[['end_x_a0', 'end_y_a0']]
    speed = passes[['speed_x_a02', 'speed_y_a02']]
    return freezeFrame, startLoc, endLoc, speed

def getTeamDictGks(match_path):
    with open(match_path, 'r') as file:
        match_data = json.load(file)
    team_dict = {player['id'] : player['team_id'] for player in match_data['players']}
    gks = [player['id'] for player in match_data['players'] if player["player_role"]['id'] == 0]
    return team_dict, gks
def get_player_xy(row):
    for player in row['player_data']:
        if player['player_id'] == row['player_id']:
            return [player['x'], player['y']]
    return [None, None] # or np.nan
def compute_distances(row):
    a = np.array(row['ball'])
    b = np.array(row['passer_loc'])
    return np.linalg.norm(a - b).tolist()
def preProcessPass(events, tracking):
    passes = events[events["end_type_id"] == 1][['match_id','event_id', 'period', 'frame_start', 'frame_end',  'attacking_side', 'player_id', 'team_id',"player_targeted_id"]]
    tracking_dropped = tracking.drop(columns = ["period"])#hopefully only have to merge on frame, not period too
    passes = pd.merge(passes, tracking_dropped, left_on = "frame_end", right_on = "frame")
    passes['ball'] = passes['ball_data'].apply(lambda x: [x['x'], x['y']])
    passes['passer_loc'] = passes.apply(lambda x: get_player_xy(x), axis = 1)
    passes['distances'] = passes.apply(lambda x: compute_distances(x), axis=1)
    return passes[passes['distances'] < 5]#cutting out distance issues



def main():
    tqdm.pandas()
    input_dir = "../../../../rdf/sp161/shared/soccer-decision-making/steffen/733681_Germany_Spain/"
    output_dir = "../../../../rdf/sp161/shared/soccer-decision-making/SkillCorner/features"
    dir_id = input_dir.rstrip('/').split('/')[-1].split('_')[0]
    sc_events = pd.read_csv(f"{input_dir}/{dir_id}_dynamic_events.csv")
    tracking = pd.read_json(f"{input_dir}{dir_id}_tracking_extrapolated.jsonl", lines = True)
    team_dict, gks = getTeamDictGks(f"{input_dir}/{dir_id}_match.json")
    passes = preProcessPass(sc_events, tracking)
    ffs, starts, ends, speeds = getAllSkillCornerFF(tracking, passes, gks, team_dict)
    idxs = starts.index
    ffs.to_frame().to_parquet(f"{output_dir}/x_freeze_frame_360.parquet")
    starts.to_parquet(f"{output_dir}/x_startlocation.parquet")
    ends.to_parquet(f"{output_dir}/x_endlocation.parquet")
    speeds.to_parquet(f"{output_dir}/x_speed.parquet")
    getHawkeyeFeats.getDummyLabels(output_dir, idxs)

if __name__ == "__main__": main()