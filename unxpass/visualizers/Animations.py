import numpy as np
import matplotlib.pyplot as plt
import mplsoccer
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import unxpass.load_xml
import pandas as pd
from unxpass.visualizers import plotPlays
def next_different_value(series):
    """
    Gets next different value in a series.
    """
    next_diff = []
    for i in range(len(series)):
        current_value = series.iloc[i]
        next_value = series.iloc[i+1:].loc[series.iloc[i+1:] != current_value].head(1)
        next_diff.append(next_value.iloc[0] if not next_value.empty else None)
    return next_diff
def animationHelper(game_id, rdfPath):
    trackingpath = f"{rdfPath}/zipped_tracking/zip_output/{game_id}.xml"
    eventpath = f"{rdfPath}//KPI_Merged_all/KPI_MGD_{game_id}.csv"
    event = unxpass.load_xml.load_csv_event(eventpath)
    tracking = unxpass.load_xml.load_tracking(trackingpath)
    return event, tracking
def get_animation_from_raw(event_id, framerate, events, tracking, custom_frames = None, show = True, add_frames = None, frameskip = 1, title = None):
    """
    Generate play animation from raw Bundesliga tracking data
    """
    events['NEXT_FRAME'] = next_different_value(events.sort_values(by = "FRAME_NUMBER")["FRAME_NUMBER"])

    fig, ax = plt.subplots(figsize=(10, 7))
    if title is not None:
        fig.suptitle(title)
    pitch = mplsoccer.Pitch(pitch_type='custom', 
                            half=False,  # Show the full pitch
                            pitch_length=105, 
                            pitch_width=68, 
                            goal_type='box', 
                            axis=True)
    pitch.draw(ax=ax)

    opponent_scatter = pitch.scatter([], [], c="r", s=30, ax=ax, marker="o")
    teammate_scatter = pitch.scatter([], [], c="b", s=30, ax=ax, marker="o")
    actor_scatter = pitch.scatter([], [], c="w", ec="k", s=20, ax=ax)
    def update(frame_num):

        teammate, opponent, actor = get_player_locations_byframe(frame_num, event_id, events, tracking)
        
        opponent_scatter.set_offsets(np.c_[opponent.x, opponent.y])
        teammate_scatter.set_offsets(np.c_[teammate.x, teammate.y])
        actor_scatter.set_offsets(np.c_[actor.x, actor.y])
        
        ax.set_title(f"Original Coords, Frame: {frame_num}")
    event = events[events["EVENT_ID"] == event_id].reset_index(drop=True).loc[0]
    end = event["NEXT_FRAME"]
    start = event["FRAME_NUMBER"]
    if start >= end:
        end = int(end) + 70
    if add_frames:
        end = int(start) + add_frames
    if custom_frames:
        frame_numbers = custom_frames
    else:
        frame_numbers = range(int(start), int(end), frameskip) 
    anim = FuncAnimation(fig, update, frames=frame_numbers, interval=framerate) 
    if show:
        
        return HTML(anim.to_jshtml())
    return anim

def get_player_locations_byframe(frame_num, event_id, events, tracking):
    team = events[events['EVENT_ID'] == event_id]["CUID1"].unique()[0]
    tracking_frame = tracking[tracking["N"] == str(frame_num)]
    teammate = tracking_frame[tracking_frame["TeamId"] == team].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    teammate["x"] = teammate["x"].str.replace(",", ".").astype(float) + 52.5
    teammate["y"] = teammate["y"].str.replace(",", ".").astype(float) + 34
    opponent = tracking_frame[(tracking_frame["TeamId"] != team) & (tracking_frame['TeamId'] != 'BALL')].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    opponent["x"] = opponent["x"].str.replace(",", ".").astype(float) + 52.5
    opponent["y"] = opponent["y"].str.replace(",", ".").astype(float) + 34
    actor = tracking_frame[tracking_frame["TeamId"] == "BALL"].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    actor["x"] = actor["x"].str.replace(",", ".").astype(float) + 52.5
    actor["y"] = actor["y"].str.replace(",", ".").astype(float) + 34
    return teammate, opponent, actor


import numpy as np
import matplotlib.pyplot as plt
import mplsoccer
def get_player_locations(event_id, events, tracking):
    event = events[events["EVENT_ID"] == event_id].reset_index(drop = True).loc[0]
    frame_num = str(event["FRAME_NUMBER"])
    player = event["PUID1"]
    team = event["CUID1"]
    tracking_frame = tracking[tracking["N"] == frame_num]
    teammate = tracking_frame[tracking_frame["TeamId"] == team].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    teammate["x"] = teammate["x"].str.replace(",", ".").astype(float) + 52.5
    teammate["y"] = teammate["y"].str.replace(",", ".").astype(float) + 34
    opponent = tracking_frame[(tracking_frame["TeamId"] != team) & (tracking_frame["TeamId"] != "BALL")].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    opponent["x"] = opponent["x"].str.replace(",", ".").astype(float) + 52.5
    opponent["y"] = opponent["y"].str.replace(",", ".").astype(float) + 34
    actor = tracking_frame[tracking_frame["PersonId"] == player].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    actor["x"] = actor["x"].str.replace(",", ".").astype(float) + 52.5
    actor["y"] = actor["y"].str.replace(",", ".").astype(float) + 34
    ball = tracking_frame[tracking_frame["TeamId"] == "BALL"].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    ball["x"] = ball["x"].str.replace(",", ".").astype(float) + 52.5
    ball["y"] = ball["y"].str.replace(",", ".").astype(float) + 34
    return teammate, opponent, actor, ball


def animate_actions(db, game_id, action_ids, surfaces=None, interval=500, show = True):
    """
    Animate actions from a game using the provided database.(legacy)
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    p = mplsoccer.Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
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
    ani = FuncAnimation(
        fig, update, frames=len(action_ids), interval=interval, repeat=False
    )

    # Display the animation
    plt.show()
    if show:
        return HTML(ani.to_jshtml())
    return ani
def get_action_ids(uuid, game_id):
    test = db.actions(game_id = game_id)
    return test[test['original_event_id'].fillna('').str.contains(uuid)].index.get_level_values('action_id').tolist()[1:]


def animate_surfaces(freeze_frame_df, start_df, speed_df, action_tuples, 
                               surfaces=None, surface_kwargs=None, log=False, interval=250, title=None, playerOnly = False, modelType = None):
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
        plotPlays.visualize_parquet_animation(
            freeze_frame_df,
            start_df,
            speed_df,
            action_tuple,
            title=f"{title} | {frame}",
            surfaces=surfaces,
            surface_kwargs=surface_kwargs,
            ax=ax,
            log=log,
            playerOnly=playerOnly,
            modelType=modelType
        )

    anim = FuncAnimation(fig, update, frames=len(action_tuples), interval=interval, repeat=False)
    plt.close(fig)
    return anim
    
def getSurfaceAnimation(index, game_id, sequence, surfaces, freeze_frame, start, speed, log = False, title = None, numFrames = 75, playerOnly = False, modelType = None):
    play_surfaces = {k:v for k,v in surfaces[game_id].items() if k.split("-")[0] == str(index)}
    action_tuples = [(game_id, key) for key in play_surfaces.keys()][0:numFrames]
    animation = animate_surfaces(freeze_frame, start, speed, action_tuples, surfaces = surfaces, surface_kwargs = {"interpolation":"bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}, log = log, title = title, playerOnly = playerOnly, modelType = modelType)
    return animation