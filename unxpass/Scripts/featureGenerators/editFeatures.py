import os
import pandas as pd 
from pathlib import Path
import numpy as np
from unxpass.Scripts.helperScripts import checkSecond
def point_to_segment_distance(P, A, B):
    """
    Compute the shortest distance from point P to line segment AB.
    
    P: np.array - the point
    A: np.array - segment start point
    B: np.array - segment end point

    Returns: float - distance from P to closest point on segment
             np.array - the closest point on the segment
    """
    P, A, B = np.array(P), np.array(A), np.array(B)
    AB = B - A
    AP = P - A
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        # A and B are the same point
        return np.linalg.norm(P - A), A
    # Compute the projection scalar of AP onto AB
    t = np.dot(AP, AB) / AB_squared
    t_clamped = np.clip(t, 0, 1)
    # Closest point on the segment
    closest_point = A + t_clamped * AB
    distance = np.linalg.norm(P - closest_point)
    return distance, closest_point
import timeit
def getClosestPlayer_vec(row, freeze_frame, start, end):
    idx = row
    ff = freeze_frame.loc[idx, "freeze_frame_360_a0"]
    recipient = [player for player in ff if player['recipient'] and player['teammate']]
    #if have a recipient that is a teammate
    if len(recipient) == 1:
        recipient = recipient[0]
        return [recipient['x'], recipient['y']]
    #if no recipient - 
    start_t = start.loc[idx]
    start_tuple = start_t[["start_x_a0", "start_y_a0"]].values
    end_t = end.loc[idx]
    end_tuple = end_t[["end_x_a0", "end_y_a0"]].values
    locs = {player['player'] : [player['x'], player['y']] for player in ff if player['teammate'] == True and player['actor'] == False}
    distances = {player: point_to_segment_distance(locs[player], start_tuple, end_tuple)[0] for player in locs}
    closest_player = min(distances, key=distances.get)
    return locs[closest_player]
from tqdm import tqdm
def getClosestPlayer(freeze_frame, start, end):
    idxs = end.index
    new_end = end.copy()
    for idx in tqdm(idxs):
        #this is somehow faster than a simply apply, idk.
        closest_end_locs = getClosestPlayer_vec(idx, freeze_frame, start, end)
        new_end.at[idx, "end_x_a0"] = closest_end_locs[0]
        new_end.at[idx, "end_y_a0"] = closest_end_locs[1]
    return new_end

def getOutlierIdx(speed_path):
    speeds = pd.read_parquet(speed_path)
    speeds['numoutlier'] = speeds.apply(lambda x: check_speeds(x['freeze_frame_360_a0']), axis = 1)
    return speeds[speeds['numoutlier'] > 0].index
def getSuccessIdx(success):
    return success[success['success']].index
def check_speeds(frame):
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
def combineIdx(idxs1, idxs2):
    idxs = set(idxs1).union(set(idxs2))
    return list(idxs)
def main():
    input_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_filtered"
    output_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_subset"
    startloc = pd.read_parquet(f"{input_dir}/x_startlocation.parquet")
    randomIdxs = startloc.sample(100).index
    #endloc = pd.read_parquet(f"{input_dir}/x_endlocation.parquet")
    #success = pd.read_parquet(f"{input_dir}/y_success.parquet")
    #distanceIdx = getDistanceThreshold(startloc, endloc, success, 0.5)#removes blocked passes
    #outlieridx = getOutlierIdx(f"{input_dir}/x_freeze_frame_360.parquet") #outlier idx
    #allidxs = combineIdx(distanceIdx, outlieridx)
    #allidxs = getSuccessIdx(success)
    #sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
    #hawkeyeEvents
    #hawkeye_events = os.listdir("/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/events")
    #trimIdxs = checkSecond.trimSecondIdx(sequences, hawkeye_events)
    getIdxs(input_dir, output_dir, idxs = randomIdxs, include = True)

    #replace end locations
    #print("Replacing End Locations...")
    #endlocs = pd.read_parquet(f"{output_dir}/x_endlocation.parquet")
    #new_ff = pd.read_parquet(f"{output_dir}/x_freeze_frame_360.parquet")
    #new_start = pd.read_parquet(f"{output_dir}/x_startlocation.parquet")
    #new_end = getClosestPlayer(new_ff, new_start, endlocs)
    #new_end.to_parquet(f"{output_dir}/x_endlocation.parquet")
if __name__ == '__main__': main()