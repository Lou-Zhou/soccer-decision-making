#edits features based on needs(e.g. removing outliers, replacing end location)
import os
import pandas as pd 
import numpy as np
import math
from tqdm import tqdm
def point_to_ray_distance(P, A, B):
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    AB = B - A
    AP = P - A

    projection_length = np.dot(AP, AB)

    if projection_length < 0:
        return np.linalg.norm(AP)
    else:
        AB_unit = AB / np.linalg.norm(AB)
        projection = projection_length * AB_unit / np.linalg.norm(AB)**2
        closest_point = A + projection_length * AB_unit / np.linalg.norm(AB)
        return np.linalg.norm(P - closest_point)

def getClosestTeammate(freeze_frame, start_loc):
    frame = freeze_frame["freeze_frame_360_a0"]
    start_x =start_loc['start_x_a0']
    start_y = start_loc['start_y_a0']
    minDist = 1e20
    minPlayer = {}
    for player in frame:
        if player['teammate'] and not player['actor']:
            p_x = player['x']
            p_y = player['y']
            distance = math.dist([p_x, p_y], [start_x, start_y])
            if distance < minDist:
                minPlayer = player
                minDist = distance
    return minPlayer, minDist
def mapClosestPlayer(row, ffdf, startdf, enddf):
    """
    For xG, returns 1 xG if pass is to closest, 0 else
    """
    idx = row.name
    freeze_frame = ffdf.loc[idx]
    start = startdf.loc[idx]
    closestTeammate = getClosestTeammate(freeze_frame, start)
    closest_x = closestTeammate[0]['x']
    closest_y = closestTeammate[0]['y']
    end_x = enddf.loc[idx]['end_x_a0']
    end_y = enddf.loc[idx]['end_y_a0']
    if end_x == closest_x and end_y == closest_y:
        return 1
    else:
        return 0

def point_to_segment_distance(P, A, B):
    P, A, B = np.array(P), np.array(A), np.array(B)
    AB = B - A
    AP = P - A
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return np.linalg.norm(P - A), A
    t = np.dot(AP, AB) / AB_squared
    t_clamped = np.clip(t, 0, 1)
    closest_point = A + t_clamped * AB
    distance = np.linalg.norm(P - closest_point)
    return distance, closest_point
def getClosestPlayer_vec(row, freeze_frame, start, end):
    """
    For a play, determine the closest player to the ray formed by start and end
    """
    idx = row
    ff = freeze_frame.loc[idx, "freeze_frame_360_a0"]
    recipient = [player for player in ff if player['recipient'] and player['teammate']]
    #if have a recipient that is a teammate
    if len(recipient) == 1:
        recipient = recipient[0]
        return [recipient['x'], recipient['y']]
    #if no recipient then c
    start_t = start.loc[idx]
    start_tuple = start_t[["start_x_a0", "start_y_a0"]].values
    end_t = end.loc[idx]
    end_tuple = end_t[["end_x_a0", "end_y_a0"]].values
    locs = {player['player'] : [player['x'], player['y']] for player in ff if player['teammate'] == True and player['actor'] == False}
    distances = {player: point_to_ray_distance(locs[player], start_tuple, end_tuple) for player in locs}
    closest_player = min(distances, key=distances.get)
    return locs[closest_player]

def checkRayDirection(P, A, B):
    """
    Checks direction of the ray, ensures that closest player is in that same direction
    """
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)
    
    AB = B - A
    AP = P - A

    dot_product = np.dot(AB, AP)

    return dot_product > 0
def checkDirections(start, og_end, recipient_end):
    """
    Check if the direction of the pass is in the same general direction of the ray, if not, return the index of the pass
    """
    weirdDirect = []
    for idx in tqdm(recipient_end.index):
        start_play = start.loc[idx,["start_x_a0", "start_y_a0"]]
        end_play = og_end.loc[idx,["end_x_a0", "end_y_a0"]]
        recipient_end_play = recipient_end.loc[idx,["end_x_a0", "end_y_a0"]]
        if not checkRayDirection(recipient_end_play, start_play, end_play):
            weirdDirect.append(idx)
    return weirdDirect

def getClosestPlayer(freeze_frame, start, end):
    idxs = end.index
    new_end = end.copy()
    for idx in tqdm(idxs):
        #this is faster than a simple apply, idk.
        closest_end_locs = getClosestPlayer_vec(idx, freeze_frame, start, end)
        new_end.at[idx, "end_x_a0"] = closest_end_locs[0]
        new_end.at[idx, "end_y_a0"] = closest_end_locs[1]
    return new_end
def getNoChange(start_df, end_df):
    """
    When using the 10 frames from method, there exist some improper frames where the ball does not move within
    the 10 frames or the ball moves very little. This function checks for those instances and returns the index of those frames.
    """
    allLocs = pd.merge(start_df, end_df, left_index = True, right_index = True)
    allLocs['distance'] = np.sqrt((allLocs['end_x_a0'] - allLocs["start_x_a0"]) ** 2 + (allLocs['end_y_a0'] - allLocs["start_y_a0"])**2)
    smallDistance = allLocs[allLocs['distance']  == 0]
    return smallDistance.index
def getOutlierIdx(speed_path):
    """
    Gets the indices of outlier speeds
    """
    speeds = pd.read_parquet(speed_path)
    speeds['numoutlier'] = speeds.apply(lambda x: check_speeds(x['freeze_frame_360_a0']), axis = 1)
    return speeds[speeds['numoutlier'] > 0].index
def getSuccessIdx(success):
    return success[success['success']].index
def check_speeds(frame):
    """
    Determines if speeds are above 11 m/s - considered outliers
    """
    speed_list = [abs(player['x_velo']) > 11 or player['y_velo'] > 11 for player in frame]
    return sum(speed_list)
def getIdxs(dir_path, output_path, idxs, include = False, random_subset = None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in  os.listdir(dir_path):
        print(f"Editing {file}")
        if include:
            parquet = pd.read_parquet(f"{dir_path}/{file}").loc[idxs]
        else:
            parquet = pd.read_parquet(f"{dir_path}/{file}")
            parquet = parquet[~parquet.index.isin(idxs)]
            if random_subset is not None:
                parquet = parquet.sample(n = random_subset, random_state = 42)
        parquet.to_parquet(f"{output_path}/{file}")
def getDistanceThreshold(start_df, end_df, success, threshold):
    #get unsuccessful passes under certain threshold
    distances = np.sqrt(
        (end_df['end_x_a0'].values - start_df['start_x_a0'].values) ** 2 +
        (end_df['end_y_a0'].values - start_df['start_y_a0'].values) ** 2
    )
    condition = success['success'].values | (distances > threshold)
    return start_df.index[~condition]
def getBlocked(end_df):
    """
    getBuliFeats marks blocked passes(where frame of interception is within 10 frames of the start) with end locations as -10000, get these indices
    """
    blocked = end_df[end_df['end_x_a0'] < -1000].index
    return blocked
def sanityCheck(start_df, end_df):
    """
    Sanity checks for impossible passes - could exist when problems with tracking / event data
    If outside of 10 yrd bound of pitch, drop
    """
    start_off = start_df[(start_df['start_x_a0'] > 110) | (start_df['start_x_a0'] < -10) | (start_df['start_y_a0'] > 78) | (start_df['start_y_a0'] < -10)].index
    end_off = end_df[(end_df['end_x_a0'] > 110) | (end_df['end_x_a0'] < -10) | (end_df['end_y_a0'] > 78) | (end_df['end_y_a0'] < -10)].index
    return combineIdx(start_off, end_off)
def combineIdx(idxs1, idxs2):
    idxs = set(idxs1).union(set(idxs2))
    return list(idxs)
def fillStartLocs(group, col):
    ref_value = group.loc[group['frames_from'] == "0", col]
    if not ref_value.empty:
        group[col] = ref_value.iloc[0]
    return group
def freezeBall(startLoc):
    """
    Freezes the ball location as requested in Hawkeye Data
    """
    split_index = startLoc.index.get_level_values(1).str.split("-", expand=True)
    startLoc['og_action'] = [index[0] for index in split_index]
    startLoc['frames_from'] = [index[1] for index in split_index]
    startLoc = startLoc.groupby('og_action', group_keys=False).apply(lambda g: fillStartLocs(g, "start_x_a0"))
    startLoc = startLoc.groupby('og_action', group_keys=False).apply(lambda g: fillStartLocs(g, "start_y_a0"))
    return startLoc[["start_x_a0", "start_y_a0"]]
from unxpass.Scripts.visualizationScripts import getAnimations
def main():
    """
    For the Bundesliga Data, feature filtering is as follows:

    sanityCheck() - ensures that no physically impossible passes occur(occur usually from data issues) as well as
    removing blocked passes(marked as endlocation = -10000)

    noChange() - sanity checker to ensure that there is a change in start and end locs, removes weirdness
    that could occur from mismatched frames with events from the data

    getOutlierIdx() - removing frames where a player goes more than 11 m/s

    getClosestPlayer() - replacing end locations with the closest player to the ray created by the start and end locations
    
    checkDirections() - ensure that ray is in the same direction as the closest player

    getSuccess() - for value models, getting successful and unsuccessful passes

    For Hawkeye Data, feature filtering just uses freezeBall to freeze the ball to time of reception as 
    needed.

    """
    input_dir = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_threeSec"
    output_dir = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_threeSec"
    startloc = pd.read_parquet(f"{input_dir}/x_startlocation.parquet")
    
    startloc = freezeBall(startloc)
    startloc.to_parquet(f"{output_dir}/x_startlocation.parquet")

    #impossibleIdx = sanityCheck(startloc, endloc) #gets indices that are not physically possible(originating from data errors)
    #outlieridx = getOutlierIdx(f"{input_dir}/x_freeze_frame_360.parquet") #outlier idx based on speed
    #noChange = getNoChange(startloc, endloc) #sanity checker to ensure that there is a change in start and end locs
    #allidxs = combineIdx(outlieridx, noChange)
    #allidxs = combineIdx(allidxs, impossibleIdx)

    #getIdxs(input_dir, output_dir, idxs = allidxs, include = False)
    #success = pd.read_parquet(f"{output_dir}/y_success.parquet")
    #successIdx = getSuccessIdx(success)
    #getIdxs(output_dir, output_dir, idxs = successIdx, include = False)

    #replace end locations
    # print("Replacing End Locations...")
    # endlocs = pd.read_parquet(f"{output_dir}/x_endlocation.parquet")
    # new_ff = pd.read_parquet(f"{output_dir}/x_freeze_frame_360.parquet")
    # new_start = pd.read_parquet(f"{output_dir}/x_startlocation.parquet")
    # new_end = getClosestPlayer(new_ff, new_start, endlocs)
    # new_end.to_parquet(f"{output_dir}/x_endlocation.parquet")
    # directionIdx = checkDirections(new_start, endlocs, new_end)
    # getIdxs(output_dir, output_dir, idxs = directionIdx, include = False)
if __name__ == '__main__': main()