import pandas as pd
import numpy as np
def getPlayerResults(surface, freeze_frame, type):
    """
    Mutating method, given a surface and a freeze frame, extracts type
    """
    for player in freeze_frame:
        start_x, start_y = player['x'], player['y']
        fixed_x = np.rint(max(0, min(start_x, 103)))#clipping between values
        fixed_y = np.rint(max(0, min(start_y, 67)))
        x_range = [int(fixed_x - 1), int(fixed_x + 1)]
        y_range = [int(fixed_y - 1), int(fixed_y + 1)]
        if modelType == "selection":
            playerSlice = surface[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            if len(playerSlice) == 0:
                alpha = 0#player is out
            else:
                alpha = np.max(playerSlice)
        else:
            alpha = surface[int(fixed_y), int(fixed_x)]
        player[modelType] = alpha

def getAllPlayerResults(row, surfaces_map):
    """
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
def concatResults(freezeFrames, models):
    df_exploded = df.explode('results', ignore_index=True)

    df_expanded = pd.concat([
        df_exploded.drop(columns='results'),
        pd.json_normalize(df_exploded['results'])
    ], axis=1)
    keepCols = models.append("original_event", "player")
    df_filtered = df_expanded[keepCols]

    agg_df = df_filtered.groupby(['original_event', 'player'], as_index=False)[[models]].mean()
    return agg_df
def getEvaluationCriterions(freeze_frame):
    output = []
    for player in freeze_frame:
        player['value_success'] = player['value_success_offensive'] - player['value_success_defensive']
        player['value_fail'] = player['value_fail_offensive'] - player['value_fail_defensive']
        player["expected_utility"] = (metrics["success_probability"] * metrics["value_success"]) + ((1 - metrics["success_probability"]) * metrics["value_fail"])
        player["evaluation_criterion"] = player["expected_utility"] - sum(player["selection_probability"] * player["expected_utility"])
        output.append(player)
    return player
def main():
    surfaces_map = {}
    dataPath = "/home/lz80/rdf/sp161/shared/socce-decision-making/Hawkeye_Features/Hawkeye_Features_Updated_wSecond_trimmed"
    outPath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkeyeResults"
    freezeFrame = pd.read_parquet(f"{dataPath}/x_freeze_frame_360.parquet")
    dataset_test = partial(PassesDataset, path=custom_path)
    #potential worries about memory issues
    model_pass_wSpeeds = pass_selection_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/0e64f9978dd04e7cb38602143178d8ce/model', map_location = 'cpu'
    )
    )
    surfaces_map["selection"] = model_pass_wSpeeds.predict_surface(dataset_test, db = None, model_name = "sel")
    freezeFrame[["original_event", "frames_from_event", "model_outputs"]] = freezeFrame.apply(lambda row: getAllPlayerResults(row, surfaces_map), axis = 1)
    freezeFrame.to_csv(f"{outPath}/allModelOutputs.csv")





