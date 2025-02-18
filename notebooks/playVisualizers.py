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
from unxpass.datasets_custom import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success, pass_value_custom
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action, plot_action_og
from unxpass.ratings_custom import LocationPredictions

from matplotlib.backends.backend_pdf import PdfPages
def plot_single_model_output(game_id, action_id, title, surfaces, db, xg = False, log = False):
    """
    Plot single model output
    """
    if xg:
        action_id_name = f"action_{action_id}"
        surface = surfaces[action_id_name]
    else:
        surface  = surfaces[game_id][action_id]
    SAMPLE = (game_id, action_id)
    ex_action = db.actions(game_id=SAMPLE[0]).loc[SAMPLE]
    #print(surface)
    #display(ex_action.to_frame().T)
    surface_params = {"vmin":0, "vmax": 1, "cmap": "Greens"}
    fig, axs = plt.subplots(1,1, figsize=(10,8))
    fig.suptitle(title, fontsize=9)
    #print(surface[game_id][action_id])
    plot_action_og(ex_action, ax = axs, surface = surface, log = log, surface_kwargs={"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, show_action = False)
    plt.tight_layout()

def plot_single_model_comparison(game_id, action_id, title, surfaces_1, surfaces_2, db, subtitle_1 = None, subtitle_2 = None, xg = False, log = False):
    """
    Plot single model output
    """
    if xg:
        action_id_name = f"action_{action_id}"
        surface_1 = surfaces_1[action_id_name]
        surface_2 = surfaces_2[action_id_name]
    else:
        surface_1  = surfaces_1[game_id][action_id]
        surface_2  = surfaces_2[game_id][action_id]
    #print(surface_1 == surface_2)
    SAMPLE = (game_id, action_id)
    ex_action = db.actions(game_id=SAMPLE[0]).loc[SAMPLE]
    #print(surface)
    #display(ex_action.to_frame().T)
    surface_params = {"vmin":0, "vmax": 1, "cmap": "Reds"}
    fig, axs = plt.subplots(1,2, figsize=(10,8))
    fig.suptitle(title, fontsize=9)
    #print(surface[game_id][action_id])
    plot_action_og(ex_action, ax = axs[0], surface = surface_1, log = log, title = subtitle_1, surface_kwargs={"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, show_action = False)
    plot_action_og(ex_action, ax = axs[1], surface = surface_2, log = log, title = subtitle_2, surface_kwargs={"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, show_action = False)
    plt.tight_layout()

def plot_model_outputs(game_id, action_id, title, surface_pass_value_success_o, surface_pass_value_success_d, surface_pass_value_fail_o, surface_pass_value_fail_d, surface_pass_success, surface_pass_selection):
    """
    Plotting Outputs of the 6 models, using the statsbomb translated coordinates
    """
    
    SAMPLE = (game_id, action_id)
    surface_pass_value_success = np.subtract(surface_pass_value_success_o[SAMPLE[0]][SAMPLE[1]], surface_pass_value_success_d[SAMPLE[0]][SAMPLE[1]])
    surface_pass_value_fail = np.subtract(surface_pass_value_fail_o[SAMPLE[0]][SAMPLE[1]], surface_pass_value_fail_d[SAMPLE[0]][SAMPLE[1]])
    surface_utility = np.multiply(surface_pass_success[SAMPLE[0]][SAMPLE[1]], surface_pass_value_success) #+ 
    ones = np.ones((surface_pass_success[SAMPLE[0]][SAMPLE[1]].shape[0], surface_pass_success[SAMPLE[0]][SAMPLE[1]].shape[1]))
    surface_fail = np.subtract(ones, surface_pass_success[SAMPLE[0]][SAMPLE[1]])
    fail = np.multiply(surface_fail, surface_pass_value_fail)
    surface_utility = np.add(surface_utility, fail)
    ex_action = db.actions(game_id=SAMPLE[0]).loc[SAMPLE]
    
    #display(ex_action.to_frame().T)
    
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    fig.suptitle(title, fontsize=9)
    ax = plot_action(ex_action, ax=axs[0][0], surface = surface_pass_selection[SAMPLE[0]][SAMPLE[1]],
                    show_visible_area=True, show_action=True,surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"})
    ax = plot_action(ex_action, ax=axs[0][1], surface = surface_utility,
                    show_visible_area=True, show_action=True,surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"})
    ax = plot_action(ex_action, ax=axs[1][0], surface = (surface_pass_success[SAMPLE[0]][SAMPLE[1]]),
                    show_visible_area=True, show_action=True,surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"})
    ax = plot_action(ex_action, ax=axs[1][1], surface = (surface_pass_value_success),
                    show_visible_area=True, show_action=True,surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"})
    ax = plot_action(ex_action, ax=axs[2][0], surface = (surface_pass_value_fail),
                    show_visible_area=True, show_action=True,surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"})
    axs[2][1].set_visible(False)
    axs[0][0].title.set_text("Selection Plot")
    axs[0][0].title.set_fontsize(6)
    axs[0][1].title.set_text("Utility Plot")
    axs[0][1].title.set_fontsize(6)
    axs[1][0].title.set_text("Pass Success Plot")
    axs[1][0].title.set_fontsize(6)
    axs[1][1].title.set_text("Pass Value Success Plot")
    axs[1][1].title.set_fontsize(6)
    axs[2][0].title.set_text("Pass Value Fail Plot")
    axs[2][0].title.set_fontsize(6)
    plt.tight_layout()
    #plt.show()
    #return fig
def read_player(path):
    """
    Reads Hawkeye data for player
    """
    df = pd.DataFrame(pd.read_json(path, convert_dates = False, lines = True, orient = 'columns')['samples'].loc[0]['people'])
    df['pos'] = df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
    df['time'] = df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
    return df
def getPlayerMap(path):
    """
    Maps player id to team ids from Hawkeye data
    """
    df = pd.DataFrame(pd.read_json(path, convert_dates = False, lines = True, orient = 'columns')).loc[0]['details']
    return {player['id']['uefaId'] : player['teamId']['uefaId'] for player in df['players']}
def read_ball(path):
    """
    Reads Hawkeye ball data
    """
    df = pd.DataFrame(pd.read_json(path, convert_dates = False, lines = True, orient = 'columns')['samples'].loc[0])
    df['pos'] = df.apply(lambda d: d['ball']['pos'], axis = 1)
    df['time'] = df.apply(lambda d: d['ball']['time'], axis = 1)
    return df
def findclosest(test_loc, time):
    """
    Finds closest time within a hawkeye dataframe within a given time
    """
    df_sort = test_loc.iloc[(test_loc['time']-time).abs().argsort()].reset_index(drop = True).loc[0]
    new_time = df_sort['time']
    return new_time
def get_locs_from_time(playerdf, balldf, time):
    """
    Gets locaitons teammates, opponents, and ball at time
    """
    playerdf = playerdf[playerdf['time'] == time].reset_index()
    balldf = balldf[balldf['time'] == time].reset_index()
    possession = balldf.loc[0]['ball']['possession']['teamId']['uefaId']
    playerdf['position'] = playerdf['role'].apply(lambda d: d['name'])
    playerdf = playerdf[playerdf['position'].isin(['Outfielder', 'Goalkeeper'])]
    playerdf['team'] = playerdf['personId'].apply(lambda d: player_dict[d['uefaId']])
    teammates = playerdf[playerdf['team'] == possession]['pos']
    opps = playerdf[playerdf['team'] != possession]['pos']
    start = balldf['pos'].loc[0][0:2]
    return teammates, opps, start
from mplsoccer import Pitch
def plotAction(teammates, opponents, start, ax = None, title = None):
    """
    Plots game state given locations of teammates, opponents, and the ball
    """
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)
    # plot visible area
    # plot freeze frame
    p.scatter([opp[0] + 52.5 for opp in opponents], [opp[1] + 34 for opp in opponents], c="r", s=30, ax=ax, marker = "x")
    p.scatter([team[0] + 52.5 for team in teammates], [team[1] + 34 for team in teammates], c="b", s=30, ax=ax, marker = "o")
   #p.scatter(teammate_locs.x, teammate_locs.y, s=30, ax=ax, facecolors=teammate_locs.selection_probability, edgecolors=teammate_locs.selection_probability)
    p.scatter(start[0] + 52.5, start[1] + 34, c="w", ec = "k", s=20, ax=ax)
    #p.scatter(60, 40, c="w", ec = "k", s=20, ax=ax)
    if title:
        ax.set_title(title, fontsize=16)
    # plot surface
    return ax
#plotAction(team, opps, start)
def getdfs(timestamp, game_id, half, path):
    """
    Gets the player dataframe, ball dataframe within the timeframe (timestamp, timestamp + 1 second) given root directory
    """
    minute = int((timestamp // 60) + 45 * (half - 1))
    second = timestamp % 60
    added_time = minute % 45
    if (minute > 45 and half == 1) or (minute > 90 and half == 2):
        minute = minute - added_time
    ball_path = f"{path}/scrubbed.samples.ball"
    player_path = f"{path}/scrubbed.samples.centroids"
    timestamp_ball_path = os.listdir(ball_path)[0]
    timestamp_player_path = os.listdir(player_path)[0]
    time_ball_start = "_".join(re.split("_", timestamp_ball_path, maxsplit = 3)[0:3])
    time_player_start = "_".join(re.split("_", timestamp_player_path, maxsplit = 3)[0:3])
    time_ball_end = re.split("\\.", timestamp_ball_path, maxsplit = 1)[1]
    time_player_end = re.split("\\.", timestamp_player_path, maxsplit = 1)[1]
    timestr = f"{half}_{int(minute)}"
    if (minute > 45 and half == 1) or (minute > 90 and half == 2):
        timestr += f"_{added_time}"
    
    player_path = f"{player_path}/{time_player_start}_{timestr}.{time_player_end}"
    print(player_path)
    ball_path = f"{ball_path}/{time_ball_start}_{timestr}.{time_ball_end}"
    playerdf = read_player(player_path)
    #playerdf = playerdf[playerdf['time'] == second]
    balldf = read_ball(ball_path)
    #balldf = balldf[balldf['time'] == second]
    closest_second = findclosest(balldf, second)
    return playerdf, balldf, closest_second
def visualize_hawkeye(timestamp, game_id, half, path, title = None):
    """
    Top level function which visualizes from raw hawkeye data
    """
    playerdf, balldf, closest_second = getdfs(timestamp, game_id, half, path)
    teammates, opps, start = plottingHelper(playerdf, balldf, closest_second)
    plotAction(teammates, opps, start, title = title)
def getTimes(uuid):
    """
    Helper function to get the timestamp and half of a statsbomb id event
    """
    idx = sequences[sequences['id'] == uuid].index
    timestamp = sequences.loc[idx]['BallReceipt']
    half = sequences.loc[idx]['half']
    return float(timestamp), int(half)

def visualize_coords_from_feature(game_id, action_id, features, ax = None):
    """
    Visualizes a pass from the raw features data
    feats - output of model.initialize_dataset(dataset, model_name = 'sel').features
    """
    # Create a pitch with a custom origin (0, 0) at the center circle
    #start_x = start_x + 52.5
    #start_y = (start_y + 34)
    #end_x =  (end_x + 52.5)
    #end_y = (end_y + 34)
    action = features.loc[(game_id, action_id)]
    start_x = action['start_x_a0']
    start_y = action['start_y_a0']
    end_x = action['end_x_a0']
    end_y = action['end_y_a0']
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
    teammate_x = [player['x'] for player in sample_action['freeze_frame_360_a0'] if player['teammate']]
    teammate_y =[player['y'] for player in sample_action['freeze_frame_360_a0'] if player['teammate']]
    opponent_x = [player['x'] for player in sample_action['freeze_frame_360_a0'] if not player['teammate'] and not player['actor']]
    opponent_y = [player['y'] for player in sample_action['freeze_frame_360_a0'] if not player['teammate'] and not player['actor']]
    # Draw a pass arrow from start to end
    pitch.arrows(start_x, start_y, end_x, end_y, width=1, headwidth=5, color='gray', ax=ax)
    
    # Scatter the start and end points for clarity
    pitch.scatter(opponent_x, opponent_y, c="r", s=30, ax=ax, marker = "x")
    pitch.scatter(teammate_x, teammate_y, c="b", s=30, ax=ax, marker = "o")
    #pitch.scatter(actor.x, actor.y, c="w", ec = "k", s=20, ax=ax)
    pitch.scatter([start_x], [start_y], c="w", ec = "k", s=20, ax=ax)
    
    # Set labels
    ax.set_title(f"Feats, action_id = {action_id}")
    plt.legend()
    
    # Show the plot
    plt.show()

def get_action_ids(uuid, game_id):
    """
    Helper function to get all "action ids"(e.g. id-x) from a base id(id)
    """
    test = db.actions(game_id = game_id)
    return test[test['original_event_id'].fillna('').str.contains(uuid)].index.get_level_values('action_id').tolist()

def visualize_coords_from_tracking(second, player_df, ball_locs, ax = None):
    """
    Visualizes a play from the raw bundesliga data
    """
    df_sort = ball_locs.iloc[(ball_locs['time']-second).abs().argsort()].reset_index(drop = True).loc[0]
    new_time = df_sort['time']
    ball_loc = ball_locs[ball_locs['time'] == new_time].reset_index(drop = True).loc[0]['pos_xy']
    player_loc = player_df[player_df['time'] == new_time]
    player_x = player_df['pos_xy'].apply(lambda d: d[0]) + 52.5
    player_y = player_df['pos_xy'].apply(lambda d: d[1]) + 34
    # Create a pitch with a custom origin (0, 0) at the center circle
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
    
    # Draw a pass arrow from start to end
    #pitch.arrows(start_x, start_y, end_x, end_y, width=1, headwidth=5, color='gray', ax=ax)
    
    # Scatter the start and end points for clarity
    pitch.scatter(player_x, player_y, c="black", s=30, ax=ax, marker = "o")
    pitch.scatter(ball_loc[0] + 52.5, ball_loc[1] + 34, c="w", ec = "k", s=20, ax=ax)
    #pitch.scatter([start_x], [start_y], c="w", ec = "k", s=20, ax=ax)
    
    # Set labels
    ax.set_title(f"Original Coords, Second:{second}")
    plt.legend()
    
    # Show the plot
    plt.show()