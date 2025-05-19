#Generate results for all models - untested
import pandas as pd
import numpy as np
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds
def getPlayerResults(surface, freeze_frame, type):
    """
    Mutating method, given a surface and a freeze frame, extracts values from the surface
    """
    for player in freeze_frame:
        start_x, start_y = player['x'], player['y']
        clipped_x = int(np.rint(start_x))#clipping between values
        clipped_y = int(np.rint(start_y))
        x_range = [clipped_x - 2, clipped_x + 2]
        y_range = [clipped_y - 2, clipped_y + 2]
        if modelType == "selection":
            playerSlice = surface[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            if len(playerSlice) == 0:
                alpha = 0#player is out
            else:
                alpha = np.max(playerSlice)
        else:
            clipped_x = min(surface.shape[1] - 1, clipped_x)
            clipped_y = min(surface.shape[0] - 1, clipped_y)
            clipped_x = max(0, clipped_x)
            clipped_y = max(0, clipped_y)
            alpha = surface[int(clipped_y), int(clipped_x)]
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
    return ogEvent, framesFrom, freezeFrame
def aggregateFreezeFrames(df, models):
    """
    For each original_event, aggregate model scores across all freeze_frame entries,
    and return a list of dicts per event with per-player averaged values.
    """
    df_exploded = df.explode('freeze_frame', ignore_index=True)

    df_expanded = pd.concat([
        df_exploded.drop(columns='freeze_frame'),
        pd.json_normalize(df_exploded['freeze_frame'])
    ], axis=1)

    grouped = (
        df_expanded
        .groupby(['original_event', 'player'], as_index=False)[models]
        .mean()
    )
    aggregated = (
        grouped
        .groupby('original_event')
        .apply(lambda g: g.drop(columns='original_event').to_dict(orient='records'))
        .reset_index(name='aggregated_freeze_frame')
    )

    return aggregated

def getGameResults(game_id, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model):
    """
    Generates results for a game
    """
    surfaces_map = getSurfaces(success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model, game_id)
    freeze_frame = pd.read_parquet(f"{dataPath}/x_freeze_frame_360.parquet")
    freeze_frame = freeze_frame.loc[freeze_frame.index.get_level_values(0) == game_id]
    freeze_frame[["original_event", "frames_from_event", "model_outputs"]] = freeze_frame.apply(lambda row: getAllPlayerResults(row, surfaces_map), axis = 1)
    return freeze_frame


def getEvaluationCriterions(row):
    """
    Generates evaluation criterions for each player in the freeze frame
    """
    output = []
    freeze_frame = row["aggregated_freeze_frame"]
    for player in freeze_frame:
        player['value_success'] = player['value_success_offensive'] - player['value_success_defensive']
        player['value_fail'] = player['value_fail_offensive'] - player['value_fail_defensive']
        player["expected_utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
        player["evaluation_criterion"] = player["expected_utility"] - sum(player["selection_probability"] * player["expected_utility"])
        output.append(player)
    return output
def getSurfaces(success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model, game_id = None):
    """
    Generates surfaces for all models
    """
    surfaces_map = {}
    surfaces_map["selection"] = selection_model.predict_surface(dataset_test,game_id = game_id)
    surfaces_map["success"] = success_model.predict_surface(dataset_test,game_id = game_id)
    surfaces_map["value_success_offensive"] = value_success_offensive_model.predict_surface(dataset_test,game_id = game_id)
    surfaces_map["value_success_defensive"] = value_success_defensive_model.predict_surface(dataset_test,game_id = game_id)
    surfaces_map["value_fail_offensive"] = value_fail_offensive_model.predict_surface(dataset_test,game_id = game_id)
    surfaces_map["value_fail_defensive"] = value_fail_defensive_model.predict_surface(dataset_test,game_id = game_id)
    return surfaces_map
def main():
    surfaces_map = {}
    dataPath = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_oneSec_trimmed"
    outPath = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/HawkeyeResults"
    freezeFrame = pd.read_parquet(f"{dataPath}/x_freeze_frame_360.parquet")
    dataset_test = partial(PassesDataset, path=custom_path)
    #potential worries about memory issues
    selection_model = pass_selection_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/3c974bbccb0e40b8a8eeab8e91ff9821/model', map_location = 'cpu'
    )
    )

    success_model = pass_success_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/cc52081bd296451189f8ca3fb9cbbee0/model', map_location = 'cpu'
    )
    )
    value_success_offensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/x/model', map_location = 'cpu'
    ), offensive = True
    )

    value_success_defensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/x/model', map_location = 'cpu'
    ), offensive = False
    )

    value_fail_offensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/x/model', map_location = 'cpu'
    ), offensive = True
    )
    
    value_fail_defensive_model = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/x/model', map_location = 'cpu'
    ), offensive = False
    )
    dfs = []
    game_ids = freezeFrame.index.get_level_values(0).unique()
    for game_id in game_ids:
        freezeFrame = getGameResults(game_id, success_model, selection_model, value_success_offensive_model, value_success_defensive_model, value_fail_offensive_model, value_fail_defensive_model)
        dfs.append(freezeFrame)
    freezeFrame = pd.concat(dfs)
    freezeFrame.to_csv(f"{outPath}/allModelOutputs.csv")
    all_models = list(surfaces_map.keys())
    eventResults = concatResults(freezeFrame, all_models)
    eventResultswithEvaluation["aggregated_freeze_frame"] = eventResults.apply(lambda row: getEvaluationCriterions(row), axis = 1)
    eventResultswithEvaluation.to_csv(f"{outPath}/allModelOutputsAggregated.csv")






