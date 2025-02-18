#Script which animates the model outputs from within the 1 second timeframe
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action_og
from unxpass.ratings_custom import LocationPredictions
STORES_FP = Path("../stores")

#db = SQLiteDatabase(STORES_FP / "womens_1_test_time.sql")
#success_model = "7be487b7b3c4438ca1cb2404a9e413cb"
model_pass_value = pass_value_custom.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/ec44b8ef79944b10a0d87d13949a1fd3/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
model_pass_selection = pass_selection.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/04d45112c139473590b5049cb3797d0d/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
model_pass_success = pass_success.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/78b7ab86dc864858b8814fe811b8796a/model', map_location='cpu'
        #'runs:/788ec5a232af46e59ac984d50ecfc1d5/model', map_location='cpu'
    )
)
#selection: runs:/04d45112c139473590b5049cb3797d0d/model
#value: runs:/ec44b8ef79944b10a0d87d13949a1fd3/model
#success: runs:/78b7ab86dc864858b8814fe811b8796a/model
#selection: runs:/f8932cf358c34aba8621993ea5b29dfe/model real one!
path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
db = SQLiteDatabase(path)
custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye_Features"
dataset_test = partial(PassesDataset, path=custom_path)
#dataset_1 = partial(PassesDataset, path=STORES_FP / "custom" / "womens_data_1")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from mplsoccer import Pitch

def animate_actions(db, game_id, action_ids, surfaces=None, interval=500):
    """
    Animate a list of actions from a game with optional surface overlays.

    Parameters
    ----------
    db : Database instance
        Database to fetch actions.
    game_id : int
        The game ID for which to fetch actions.
    action_ids : list of int
        List of action IDs to animate.
    surfaces : list of np.array, optional
        List of surfaces (heatmaps, overlays, etc.) corresponding to each action.
    interval : int, optional
        Time interval between frames (in milliseconds).
    """
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    p.draw(ax=ax)  # Draw the pitch initially

    def update(frame_idx):
        # Clear the axis for the next frame and re-draw the pitch
        ax.clear()
        p.draw(ax=ax)
        
        # Fetch the action for the current frame
        action_id = action_ids[frame_idx]
        action = db.actions(game_id=game_id).loc[(game_id,action_id)]

        # Determine the surface to plot (if any)
        surface = surfaces[game_id][action_id] if surfaces is not None else None
        
        # Plot the action with optional surface
        plot_action_og(action, surface=surface, ax=ax,surface_kwargs={"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"})

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(action_ids), interval=interval, repeat=False
    )

    # Display the animation
    plt.show()

    return ani

# Usage example:
# db = ...  # Your database instance
# game_id = 12345
# action_ids = [1, 2, 3, 4]  # List of action IDs
# surfaces = [np.random.random((68, 105)) for _ in action_ids]  # Example list of surfaces
# animate_actions(db, game_id, action_ids, surfaces)
def get_action_ids(uuid, game_id):
    test = db.actions(game_id = game_id)
    return test[test['original_event_id'].fillna('').str.contains(uuid)].index.get_level_values('action_id').tolist()[1:]
game_id = '3835319'
action_id = '1e97c5d0-024e-419a-a2f4-ec15053d2337'
			
#print(db.actions(game_id = game_id))
print(db.games())
surfaces = model_pass_selection.predict_surface(dataset_test, db = db, model_name = "sel", game_id = game_id)
action_ids = get_action_ids(action_id, game_id)
print(action_ids)
ani = animate_actions(db, game_id, action_ids, surfaces = surfaces)
writergif = animation.PillowWriter(fps=15)
ani.save('selection_test.gif', writer=writergif)