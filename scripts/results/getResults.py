#Generate results for all models - relatively inefficient, might be worth optimizing in the future
import pandas as pd
import numpy as np
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds
from collections import defaultdict
from functools import partial
from unxpass.datasets import PassesDataset
import mlflow
from tqdm import tqdm
def surfaceToPlayer(surface,players):
    """
    Maps each player to a list of surface values closest to them.
    """
    player_ids = np.array([p['player'] for p in players if p['teammate'] and not p['actor']])
    player_coords = np.array([[p['x'], p['y']] for p in players if p['teammate'] and not p['actor']])

    height, width = surface.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height)) #builds coordinate grids
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2) #combine into a grid of coordinates


    distances = np.linalg.norm(coords[:, None, :] - player_coords[None, :, :], axis=2) 
    #compute all distances, 3d matrix of distances of all players

    #get closest distance
    closest_indices = np.argmin(distances, axis=1)
    closest_players = player_ids[closest_indices] 

    surface_values = surface.flatten()

    player_sums = defaultdict(float)
    for pid, val in zip(closest_players, surface_values):
        player_sums[pid] += val

    return dict(player_sums)

def getPlayerResults(surface, freeze_frame, modelType):
    """
    Mutating method, given a surface and a freeze frame, extracts values from the surface
    """
    if modelType == "selection":
        player_vals =  surfaceToPlayer(surface, freeze_frame)
    for player in freeze_frame:
        if not player['teammate'] or player['actor']:
            continue
        start_x, start_y = player['x'], player['y']
        clipped_x = np.clip(start_x / 105 * surface.shape[1], 0, surface.shape[1] - 1).astype(np.uint8)
        clipped_y = np.clip(start_y / 105 * surface.shape[0], 0, surface.shape[0] - 1).astype(np.uint8)
        if modelType == "selection":
            alpha = player_vals[player['player']]
        else:
            alpha = surface[clipped_y, clipped_x]
        player[modelType] = alpha

def getAllPlayerResults(row, surfaces_map):
    """
    Gets model results for all players in a freeze frame
    Surfaces_map - mapping of string describing type and dictionary of all the surfaces
    """
    index = row.name
    freezeFrame = row["freeze_frame_360_a0"]
    ogEvent = index[1].split("-")[0]
    framesFrom = index[1].split("-")[1]
    for modelType in surfaces_map:
        surface = surfaces_map[modelType][index[0]][index[1]]
        getPlayerResults(surface, freezeFrame, modelType)
    return pd.Series({"original_event":ogEvent, "frames_from_event":framesFrom, "model_outputs": freezeFrame})
import ast
def aggregateFreezeFrames(df, models):
    """
    For each original_event, aggregate model scores across all freeze_frame entries,
    and return a dataframe with per-player / event averaged values.
    """
    cols = models + ["evaluation_criterion"]
    grouped = (
        df
        .groupby(['game', 'original_event', 'player'], as_index=False)[cols]
        .mean()
    )

    return grouped

def getGameResults(dataPath, game_id, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model):
    """
    Generates results for a game
    """
    dataset = partial(PassesDataset, path=dataPath)
    surfaces_map = getSurfaces(dataset, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model, game_id)
    freeze_frame = pd.read_parquet(f"{dataPath}/x_freeze_frame_360.parquet")
    freeze_frame = freeze_frame.loc[freeze_frame.index.get_level_values(0) == game_id]
    freeze_frame[["original_event", "frames_from_event", "model_outputs"]] = freeze_frame.apply(lambda row: getAllPlayerResults(row, surfaces_map), axis = 1)
    freeze_frame = freeze_frame.explode("model_outputs", ignore_index=True)
    freeze_frame_exploded = pd.concat([
        freeze_frame.drop(columns=["model_outputs", "freeze_frame_360_a0"]),
        pd.json_normalize(freeze_frame["model_outputs"])
    ], axis=1)
    freeze_frame_exploded['game'] = game_id
    freeze_frame_exploded['value_success'] = freeze_frame_exploded['value_success_offensive'] - freeze_frame_exploded['value_success_defensive']
    freeze_frame_exploded['value_fail'] = freeze_frame_exploded['value_fail_offensive'] - freeze_frame_exploded['value_fail_defensive']
    freeze_frame_exploded['expected_utility'] = freeze_frame_exploded['success'] * freeze_frame_exploded['value_success'] + (1 - freeze_frame_exploded['success']) * freeze_frame_exploded['value_fail']
    total_eval = ( freeze_frame_exploded.groupby('original_event')
      .apply(getTotalEvaluationCriterions)
      .rename('total_eval_criterion')
    )
    freeze_frame_exploded['total_eval_criterion'] = freeze_frame_exploded['original_event'].map(total_eval)
    freeze_frame_exploded['evaluation_criterion'] = freeze_frame_exploded['expected_utility'] - freeze_frame_exploded['total_eval_criterion']
    return freeze_frame_exploded


def getTotalEvaluationCriterions(group):
    """
    Generates total evaluation criterions
    """
    onlyTeammate = group[~pd.isna(group['expected_utility'])]
    return sum(onlyTeammate['selection'] * onlyTeammate['expected_utility'])
def getSurfaces(dataset, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model, game_id = None):
    """
    Generates surfaces for all models
    """
    surfaces_map = {}
    surfaces_map["selection"] = selection_model.predict_surface(dataset,game_id = game_id)
    surfaces_map["success"] = success_model.predict_surface(dataset,game_id = game_id)
    surfaces_map["value_success_offensive"] = value_success_offensive_model.predict_surface(dataset,game_id = game_id)
    surfaces_map["value_success_defensive"] = value_success_defensive_model.predict_surface(dataset,game_id = game_id)
    surfaces_map["value_fail_offensive"] = value_fail_offensive_model.predict_surface(dataset,game_id = game_id)
    surfaces_map["value_fail_defensive"] = value_fail_defensive_model.predict_surface(dataset,game_id = game_id)
    return surfaces_map
def main():
    dataPath = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_oneSec"
    outPath = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/HawkeyeResults"
    freezeFrame = pd.read_parquet(f"{dataPath}/x_freeze_frame_360.parquet")
    #potential worries about memory issues
    selection_model = pass_selection_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/d7ce231b1e274d70b03db07ab31d8cda/model', map_location = 'cpu'
    )
    )

    success_model = pass_success_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/3c974bbccb0e40b8a8eeab8e91ff9821/model', map_location = 'cpu'
    )
    )
    value_success_offensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/4e97c61c9b6749719c2bdf3d05292a07/model', map_location = 'cpu'
    ), offensive = True
    )

    value_success_defensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/e6841372614045cd8f72c62943b3c858/model', map_location = 'cpu'
    ), offensive = False
    )

    value_fail_offensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/8d3584119cd746469e99368e04e19b03/model', map_location = 'cpu'
    ), offensive = True
    )
    
    value_fail_defensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/f73621e21a064dd4bcb199efce9b4d26/model', map_location = 'cpu'
    ), offensive = False
    )
    dfs = []
    game_ids = freezeFrame.index.get_level_values(0).unique()
    for game_id in tqdm(game_ids):
        #Done on a per-game basis due to fears of memory issues, if small, could theoretically do entire dataset(would be faster too)
       freezeFrame = getGameResults(dataPath, game_id, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model)
       dfs.append(freezeFrame)
    freezeFrame = pd.concat(dfs)
    freezeFrame.to_csv(f"{outPath}/allModelOutputs.csv")
    all_models = ["selection", "success", "value_success_offensive", "value_success_defensive", "value_fail_offensive", "value_fail_defensive"]
    eventResults = aggregateFreezeFrames(freezeFrame, all_models)
    eventResults.to_csv(f"{outPath}/allModelOutputsAggregated.csv")
if __name__ == "__main__": main()





