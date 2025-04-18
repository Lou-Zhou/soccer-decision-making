from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from scipy.ndimage import zoom
import warnings
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.withSpeeds import pass_value_speeds
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.converters import playVisualizers
from unxpass.ratings_custom import LocationPredictions
from matplotlib.backends.backend_pdf import PdfPages
import mplsoccer
def visualize_coords_from_tracking(freeze_frame, start, speed, action_tuple, title = None, surfaces = None, surface_kwargs = None, ax = None, log = False):
    ff_action = freeze_frame.loc[action_tuple, 'freeze_frame_360_a0']
    teammate_x = [player['x'] for player in ff_action if player['teammate']]
    teammate_y = [player['y'] for player in ff_action if player['teammate']]
    opponent_x = [player['x'] for player in ff_action if not player['teammate']]
    opponent_y = [player['y'] for player in ff_action if not player['teammate']]
    x_velo = [player['x_velo'] for player in ff_action]
    y_velo = [player['y_velo'] for player in ff_action]
    ball_x = start.loc[action_tuple, 'start_x_a0']
    ball_y = start.loc[action_tuple, 'start_y_a0']
    pitch = mplsoccer.pitch.Pitch(pitch_type='custom', 
                  half=False,         # Show only half the pitch (positive quadrant)
                  pitch_length=105,   # Length of the pitch (in meters)
                  pitch_width=68,     # Width of the pitch (in meters)
                  goal_type='box',
                  axis=True)          # Show axis for coordinates
    
    # Create a figure
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
    else:
        pitch.draw(ax=ax)
    for player in ff_action:
        start_x, start_y = player['x'], player['y']
        end_x, end_y = player['x'] + player['x_velo'], player['y'] + player['y_velo']
        pitch.arrows(start_x, start_y, end_x, end_y, width=1, headwidth=5, color='gray', ax=ax)
    # Scatter the start and end points for clarity
    pitch.scatter(opponent_x, opponent_y, c="r", s=30, ax=ax, marker = "x")
    pitch.scatter(teammate_x, teammate_y, c="b", s=30, ax=ax, marker = "o")
    pitch.scatter([ball_x], [ball_y], c="w", ec = "k", s=20, ax=ax)
    if surfaces is not None:
        surface = surfaces[action_tuple[0]][action_tuple[1]]
        if log:
            ax.imshow(np.log(surface), extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)
        else:
            ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)
    
    # Set labels
    ax.set_title(title)
    
    # Show the plot
    plt.show()
import unxpass.converters.playVisualizers
from unxpass.components.withSpeeds import pass_selection_speeds

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_tracking_sequence(freeze_frame_df, start_df, speed_df, action_tuples, 
                               surfaces=None, surface_kwargs=None, log=False, interval=250, title=None):
    """
    Creates an animation of player coordinates and surfaces for a sequence of actions.
    
    Parameters:
        freeze_frame_df: DataFrame containing freeze frame data
        start_df: DataFrame with starting ball positions
        speed_df: (unused but kept for compatibility)
        action_tuples: List of (index0, index1) tuples identifying each frame to plot
        surfaces: Optional 2D list of surface arrays
        surface_kwargs: Arguments to pass to imshow (e.g., cmap, vmin/vmax)
        log: Whether to apply np.log to surface
        interval: Time in ms between frames
        title_prefix: Prefix for subplot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = mplsoccer.pitch.Pitch(pitch_type='custom',
                                  half=False,
                                  pitch_length=105,
                                  pitch_width=68,
                                  goal_type='box',
                                  axis=True)
    pitch.draw(ax=ax)

    def update(frame):
        ax.clear()
        action_tuple = action_tuples[frame]
        visualize_coords_from_tracking(
            freeze_frame_df,
            start_df,
            speed_df,
            action_tuple,
            title=f"{title} | {frame}",
            surfaces=surfaces,
            surface_kwargs=surface_kwargs,
            ax=ax,
            log=log
        )

    anim = FuncAnimation(fig, update, frames=len(action_tuples), interval=interval, repeat=False)
    plt.close(fig)
    return anim
    
def getAnimation(index, game_id, sequence, surfaces, freeze_frame, start, speed, log = False, title = None):
    play_surfaces = {k:v for k,v in surfaces[game_id].items() if k.split("-")[0] == str(index)}
    action_tuples = [(game_id, key) for key in play_surfaces.keys()]
    animation = animate_tracking_sequence(freeze_frame, start, speed, action_tuples, surfaces = surfaces, surface_kwargs = {"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, log = log, title = title)
    return animation
    from pathlib import Path

def main(num_to_generate = 5, custom_game = None):
    custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Hawkeye_Features/Hawkeye_Features_Updated_wSecond"
    dataset_test = partial(PassesDataset, path=custom_path)
    model = pass_selection_speeds.SoccerMapComponent(
        model=mlflow.pytorch.load_model(
            'runs:/739d103329bf4d399fbb8d311859382a/model', map_location='cpu'
            #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
        )
    )
    sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
    surfaces = model.predict_surface(dataset_test, model_name = "sel")
    freeze_frame_df = pd.read_parquet(f"{custom_path}/x_freeze_frame_360.parquet")
    speed_df = pd.read_parquet(f"{custom_path}/x_speed.parquet")
    start_df = pd.read_parquet(f"{custom_path}/x_startlocation.parquet")
    for anim in range(num_to_generate):
        if custom_game is not None:
            game_id = custom_game
        else:
            game_id = sequences.iloc[anim]["match_id"]
        game_sequences = sequences[sequences["match_id"] == game_id]
        idx = game_sequences.iloc[anim]["index"]
        s_id = game_sequences.iloc[anim]["id"]
        
        animation = getAnimation(idx, game_id, sequences, surfaces, freeze_frame_df, start_df, speed_df, log = True, title = f"Selection Probabilities | {game_id} | {s_id}")
        animation_title = f"/home/lz80/un-xPass/unxpass/Scripts/animations/{game_id}_{idx}_sel.gif"
        animation.save(animation_title, writer='pillow', fps=1, dpi=200)
if __name__ == '__main__': main(num_to_generate = 25)  # Change the number of animations to generate as needed