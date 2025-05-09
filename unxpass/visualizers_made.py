import numpy as np
import matplotlib.pyplot as plt
import mplsoccer
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import unxpass.load_xml
import pandas as pd
def next_different_value(series):
    next_diff = []
    for i in range(len(series)):
        # Get the current value
        current_value = series.iloc[i]
        # Find the next different value
        next_value = series.iloc[i+1:].loc[series.iloc[i+1:] != current_value].head(1)
        # Append the found value or None if there is no next different value
        next_diff.append(next_value.iloc[0] if not next_value.empty else None)
    return next_diff
def animationHelper(game_id):
    trackingpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/zipped_tracking/zip_output/{game_id}.xml"
    eventpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/KPI_Merged_all/KPI_MGD_{game_id}.csv"
    event = unxpass.load_xml.load_csv_event(eventpath)
    tracking = unxpass.load_xml.load_tracking(trackingpath)
    return event, tracking
def get_animation_from_raw(event_id, framerate, events, tracking, custom_frames = None, show = True, add_frames = None, frameskip = 1, title = None):
    #events = csv, tracking = xml
    events['NEXT_FRAME'] = next_different_value(events.sort_values(by = "FRAME_NUMBER")["FRAME_NUMBER"])
    def get_player_locations_byframe(frame_num):
        team = events["CUID1"].unique()[0]
        tracking_frame = tracking[tracking["N"] == str(frame_num)]
        teammate = tracking_frame[tracking_frame["TeamId"] == team].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
        teammate["x"] = teammate["x"].str.replace(",", ".").astype(float) + 52.5
        teammate["y"] = teammate["y"].str.replace(",", ".").astype(float) + 34
        opponent = tracking_frame[(tracking_frame["TeamId"] != team) & (tracking_frame["TeamId"] != "BALL")].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
        opponent["x"] = opponent["x"].str.replace(",", ".").astype(float) + 52.5
        opponent["y"] = opponent["y"].str.replace(",", ".").astype(float) + 34
        actor = tracking_frame[tracking_frame["TeamId"] == "BALL"].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
        actor["x"] = actor["x"].str.replace(",", ".").astype(float) + 52.5
        actor["y"] = actor["y"].str.replace(",", ".").astype(float) + 34
        return teammate, opponent, actor

    # Create the figure and axis for the pitch (without rendering it yet)
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

    # Initialize scatter plot objects for the players and the ball
    opponent_scatter = pitch.scatter([], [], c="r", s=30, ax=ax, marker="o")
    teammate_scatter = pitch.scatter([], [], c="b", s=30, ax=ax, marker="o")
    actor_scatter = pitch.scatter([], [], c="w", ec="k", s=20, ax=ax)

# This function will update the plot for each frame
    def update(frame_num):
        # Get the current player locations
        teammate, opponent, actor = get_player_locations_byframe(frame_num)
        
        # Update the scatter plot data with the new positions for this frame
        opponent_scatter.set_offsets(np.c_[opponent.x, opponent.y])
        teammate_scatter.set_offsets(np.c_[teammate.x, teammate.y])
        actor_scatter.set_offsets(np.c_[actor.x, actor.y])
        
        # Set frame-specific title
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
        frame_numbers = range(int(start), int(end), frameskip)  # 13640 is not inclusive, so it goes up to 13639
    anim = FuncAnimation(fig, update, frames=frame_numbers, interval=framerate)  # Adjust the interval for speed (200ms between frames)
    print([min(frame_numbers), max(frame_numbers)])
    # Save the animation as a video file or display it inline
    # To display inline in Jupyter, use:
    if show:
        
        return HTML(anim.to_jshtml())
    return anim

def get_player_locations_byframe(frame_num, event_id, events, tracking):
    team = events[events['EVENT_ID'] == event_id]["CUID1"].unique()[0]
    tracking_frame = tracking[tracking["N"] == str(frame_num)]
    teammate = tracking_frame[tracking_frame["TeamId"] == team].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    teammate["x"] = teammate["x"].str.replace(",", ".").astype(float) + 52.5
    teammate["y"] = teammate["y"].str.replace(",", ".").astype(float) + 34
    opponent = tracking_frame[tracking_frame["TeamId"] != team].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    opponent["x"] = opponent["x"].str.replace(",", ".").astype(float) + 52.5
    opponent["y"] = opponent["y"].str.replace(",", ".").astype(float) + 34
    actor = tracking_frame[tracking_frame["TeamId"] == "BALL"].rename(columns = {"X":"x", "Y":"y"})[["x","y"]]
    actor["x"] = actor["x"].str.replace(",", ".").astype(float) + 52.5
    actor["y"] = actor["y"].str.replace(",", ".").astype(float) + 34
    return teammate, opponent, actor

def visualize_coords_from_tracking(frame_num, events, tracking, ax = None):
    teammate, opponent, actor = get_player_locations_byframe(frame_num, tracking)
    """
    Visualizes a pass on a football pitch with a custom coordinate system.
    
    Args:
    start_x (float): The x-coordinate of the pass start position (relative to the center).
    start_y (float): The y-coordinate of the pass start position (relative to the center).
    end_x (float): The x-coordinate of the pass end position (relative to the center).
    end_y (float): The y-coordinate of the pass end position (relative to the center).
    """
    # Create a pitch with a custom origin (0, 0) at the center circle
    #start_x = start_x + 52.5
    #start_y = (start_y + 34)
    #end_x =  (end_x + 52.5)
    #end_y = (end_y + 34)
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
    pitch.scatter(opponent.x, opponent.y, c="r", s=30, ax=ax, marker = "x")
    pitch.scatter(teammate.x, teammate.y, c="b", s=30, ax=ax, marker = "o")
    pitch.scatter(actor.x, actor.y, c="w", ec = "k", s=20, ax=ax)
    #pitch.scatter([start_x], [start_y], c="w", ec = "k", s=20, ax=ax)
    
    # Set labels
    ax.set_title(f"Original Coords, Frame:{frame_num}")
    plt.legend()
    
    # Show the plot
    plt.show()
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
    return teammate, opponent, actor

def visualize_pass_from_raw(event_id, events, tracking, ax = None):
    event = events[events["EVENT_ID"] == event_id].reset_index(drop = True).loc[0]
    start_x = float(event["X_EVENT"].replace(",", "."))
    start_y = float(event["Y_EVENT"].replace(",", "."))
    if pd.notna(event["XRec"]):
        end_x = float(event["XRec"].replace(",", "."))
        end_y = float(event["YRec"].replace(",", "."))
    teammate, opponent, actor = get_player_locations(event_id, events, tracking)
    """
    Visualizes a pass on a football pitch with a custom coordinate system.
    
    Args:
    start_x (float): The x-coordinate of the pass start position (relative to the center).
    start_y (float): The y-coordinate of the pass start position (relative to the center).
    end_x (float): The x-coordinate of the pass end position (relative to the center).
    end_y (float): The y-coordinate of the pass end position (relative to the center).
    """
    # Create a pitch with a custom origin (0, 0) at the center circle
    start_x = start_x + 52.5
    start_y = (start_y + 34)
    if pd.notna(event["XRec"]):
        end_x =  (end_x + 52.5)
        end_y = (end_y + 34)
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
    if pd.notna(event["XRec"]):
        pitch.arrows(start_x, start_y, end_x, end_y, width=1, headwidth=5, color='gray', ax=ax)
    # Scatter the start and end points for clarity
    pitch.scatter(opponent.x, opponent.y, c="r", s=30, ax=ax, marker = "x")
    pitch.scatter(teammate.x, teammate.y, c="b", s=30, ax=ax, marker = "o")
    pitch.scatter(actor.x, actor.y, c="g", s=30, ax=ax, marker = "o")
    pitch.scatter([start_x], [start_y], c="w", ec = "k", s=20, ax=ax)
    
    # Set labels
    ax.set_title(f"Original Coords, id:{event_id}")
    plt.legend()
    
    # Show the plot
    plt.show()
from unxpass.visualization import plot_action, plot_action_og
def plot_test_action_from_db(id, db_actions, ax = None):
    a = db_actions[db_actions["original_event_id"] == str(id) + ".0"].index
    plot_action(db_actions.loc[a[0]], title = f"Transformed Coords, id: {id}", ax = ax)

def animate_actions(db, game_id, action_ids, surfaces=None, interval=500, show = True):
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