#Generate gifs of surfaces
import pandas as pd
import numpy as np
from functools import partial
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds
from unxpass.datasets import PassesDataset
import mlflow
from unxpass.visualizers import Animations



def main(num_to_generate = 5, custom_game = None):
    custom_path = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_tenSecPrior"
    dataset_test = partial(PassesDataset, path=custom_path)
    model = pass_success_speeds.SoccerMapComponent(
        model=mlflow.pytorch.load_model(
            'runs:/3c974bbccb0e40b8a8eeab8e91ff9821/model', map_location='cpu'
            #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
        )
    )
    modelType = "success"
    sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv", delimiter = ";")
    sample_game = sequences.iloc[0]["match_id"]
    surfaces = model.predict_surface(dataset_test, game_id = sample_game)
    freeze_frame_df = pd.read_parquet(f"{custom_path}/x_freeze_frame_360.parquet")
    speed_df = pd.read_parquet(f"{custom_path}/x_speed.parquet")
    start_df = pd.read_parquet(f"{custom_path}/x_startlocation.parquet")
    for anim in range(num_to_generate):
        if sample_game is not None:
            game_id = sample_game
        else:
            game_id = sequences.iloc[anim]["match_id"]
        game_sequences = sequences[sequences["match_id"] == game_id]
        idx = game_sequences.iloc[anim]["index"]
        s_id = game_sequences.iloc[anim]["id"]
        animation = Animations.getSurfaceAnimation(idx, game_id, sequences, surfaces, freeze_frame_df, start_df, speed_df, log = True, title = f"Selection Probabilities | {game_id} | {s_id} 10Sec", numFrames = 251, playerOnly = True, modelType = modelType)
        animation_title = f"../visualizations/animations/animation_{game_id}_{s_id}_10Sec.gif"
        animation.save(animation_title, writer='pillow', fps=10, dpi=200)
if __name__ == '__main__': main(num_to_generate = 5)  # Change the number of animations to generate as needed