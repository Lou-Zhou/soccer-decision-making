#weirdos: 18453000001058
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
from unxpass import visualizers_made
def getAngleFrames(sample_pass, allAngles):
    start_frame = sample_pass['FRAME_NUMBER']
    event_id = sample_pass['EVENT_ID']
    angleStartFrame = allAngles[(allAngles['pass_start_angle'] > 20) & (allAngles['N'] > start_frame + 10)]
    angleStartFrame = angleStartFrame.loc[angleStartFrame['N'].idxmin()]['N']
    return angleStartFrame
#visualize_coords_from_tracking()
def getAngleChangeFromStart(tracking, pass_start_map, fb = 10):
    """
    Compute angle change between ball direction at pass start and at each frame.

    Parameters:
    - tracking: DataFrame with ball tracking data (must contain 'N', 'X', 'Y', 'GameSection', 'TeamId')
    - pass_start_map: dict or Series mapping frame N to pass start frame N0

    Returns:
    - DataFrame with X, Y, angle_change, N, and GameSection
    """
    ball_data_df = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball_data_df['N'] = ball_data_df['N'].astype(float)
    ball_data_df['X'] = ball_data_df['X'].astype(float)
    ball_data_df['Y'] = ball_data_df['Y'].astype(float)
    ball_data_df = ball_data_df.sort_values(by='N')

    # Compute dx, dy for current frame
    ball_data_df['dx'] = ball_data_df['X'] - ball_data_df.groupby('GameSection')['X'].shift(10)
    ball_data_df['dy'] = ball_data_df['Y'] - ball_data_df.groupby('GameSection')['Y'].shift(10)
    ball_data_df['magnitude'] = np.sqrt(ball_data_df['dx']**2 + ball_data_df['dy']**2)

    # Map each N to its pass start frame
    ball_data_df['pass_start_N'] = ball_data_df['N'].map(pass_start_map)
    
    ball = tracking[tracking['TeamId'] == 'BALL'].copy()
    ball[['N', 'X', 'Y']] = ball[['N', 'X', 'Y']].astype(float)
    ball = ball.sort_values('N')
    ball = ball.drop_duplicates(subset='N')  # Ensure no duplicate frames
    ball_indexed = ball.set_index('N')

    # Get start and end positions in one go
    start_pos = ball_indexed.reindex(ball_data_df['pass_start_N'])
    end_pos = ball_indexed.reindex(ball_data_df['pass_start_N'] + fb)

    # Compute vector difference
    start_dxdy = end_pos[['X', 'Y']].values - start_pos[['X', 'Y']].values

    # Assign to new columns
    ball_data_df[['start_dx', 'start_dy']] = start_dxdy
    ball_data_df['start_magnitude'] = np.hypot(ball_data_df['start_dx'], ball_data_df['start_dy'])
    
    # Compute dot product between current and pass start vectors
    ball_data_df['dot_product'] = (ball_data_df['dx'] * ball_data_df['start_dx'] +
                                   ball_data_df['dy'] * ball_data_df['start_dy'])

    # Calculate angle in degrees
    cosine_values = ball_data_df['dot_product'] / (ball_data_df['magnitude'] * ball_data_df['start_magnitude'])
    cosine_values = np.clip(cosine_values, -1, 1)
    ball_data_df['angle_change'] = np.degrees(np.arccos(cosine_values))

    # Clean up infinities and NaNs
    ball_data_df['angle_change'].replace([np.inf, -np.inf], np.nan, inplace=True)
    ball_data_df['angle_change'].fillna(0, inplace=True)

    return ball_data_df[['X', 'Y', 'angle_change', 'N','pass_start_N', 'GameSection', 'start_dx', 'start_dy', 'start_magnitude']]

from unxpass import visualizers_made
from tqdm import tqdm
from unxpass.Scripts.featureGenerators import get_Hawkeye_Bundesliga_Features
import json
import ast
def main():
    errors = []
    with open('/home/lz80/un-xPass/unxpass/Scripts/index_dict.json', 'r') as f:
        index_dict = json.load(f)
    for game_id in tqdm(list(index_dict.keys())):
        event, tracking = visualizers_made.animationHelper(game_id)
        og_event = event.copy()
        og_tracking = tracking.copy()#some mutation problems - bandaid fix
        events = index_dict[game_id]
        events = list(map(float, events))
        intercepts = event[event["EVENT_ID"].isin(events)]
        #intercepts = event[(event['EVALUATION'] == 'unsuccessful') & (~pd.isna(event['PUID2']))][["EVENT_ID", "FRAME_NUMBER"]].copy()
        #event_map = add_speed.map_n_event(tracking, event)
        #angles_pass_start = getAngleChangeFromStart(tracking, event_map)[['N', 'angle_change']]
        #angles_pass_start.rename(columns = {'angle_change': 'pass_start_angle'}, inplace = True)
        #receipts = add_speed.getReceipts(tracking, event)
        for idx, row in tqdm(intercepts.loc[0:50].iterrows()):
            try:
                event_id = row['EVENT_ID']
                frame = row['FRAME_NUMBER']
                #event = receipts[(receipts['EVENT_ID'] == event_id)]
                #eventReceipt = receipts[(receipts['EVENT_ID'] == event_id) & (~pd.isna(receipts['RECFRM']))].iloc[0]["RECFRM"]
                #angleStartFrame = getAngleFrames(row, angles_pass_start)
                #angleStartFrame = int(angleStartFrame)
                #eventReceipt = int(eventReceipt)
                #custom_frames = list(range(int(frame), max([int(frame) + 75, angleStartFrame + 1, eventReceipt + 1])))
                custom_frames = range(int(frame), int(frame) + 75)
                #custom_frames = custom_frames + [angleStartFrame, eventReceipt] * 5
                #custom_frames.sort()
                #title = f"Event {event_id} | Tracking {angleStartFrame} | Event {eventReceipt}"
                title = f"Game {game_id} | Event {event_id}"
                animation = visualizers_made.get_animation_from_raw(event_id, 1, og_event, og_tracking,custom_frames = custom_frames, show = False, add_frames = None, frameskip = 1, title = title)
                animation_title = f"/home/lz80/un-xPass/notebooks/playViz/{event_id}.gif"
                animation.save(animation_title, writer='pillow', fps=5, dpi=200)
            except:
                print("Error")
                event_id = row['EVENT_ID']
                errors.append(event_id)
if __name__ == "__main__":
    main()