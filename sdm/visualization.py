import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
from tqdm import tqdm
import mlflow
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import mplsoccer
from unxpass.datasets import PassesDataset
from unxpass.components.withSpeeds import pass_selection_speeds, pass_success_speeds, pass_value_speeds

from . import animation, path_data, results

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

def visualize_surface(component,
                      freeze_frame,
                      surface,
                      start,
                      end=None,
                      ball_speed=None,
                      title=None,
                      ax=None,
                      show_pass=True,
                      show_surface=False,
                      surface_kwargs=None,
                      log=False):
    """
    Visualize a single soccer game state on a pitch with players, ball trajectory, 
    and an optional model-generated surface overlay.

    This function plots the game state from tracking and event data (stored in parquet files),
    along with predicted surfaces (e.g., pass success or selection probabilities).
    Players, their movement vectors, ball start/end locations, and ball velocities
    are drawn on top of the pitch. If a surface is provided, it is visualized as a heatmap.

    Parameters
    ----------
    component : {'selection', 'success', 'value'}
        Determines how surface values are mapped:
        - 'selection': player-specific values are extracted from the surface and used for alpha transparency.
        - 'value': surface values are plotted directly.
    freeze_frame : pandas.Series or dict-like
        Freeze-frame data for the action (e.g., `freeze_frame_360_a0` row from parquet).
        Contains player positions, velocities, and metadata (teammate, actor, etc.).
    surface : ndarray
        2D array representing the model surface (e.g., probabilities).
    start : pandas.Series or dict-like
        Start location of the ball for this action (e.g., `x_startlocation` parquet row).
    end : pandas.Series or dict-like, optional
        End location of the ball for this action (e.g., `x_endlocation` parquet row).
        If provided, a black arrow will be drawn from start to end.
    ball_speed : pandas.Series or dict-like, optional
        Ball velocity components (e.g., `x_speed` parquet row).
        If provided, a velocity arrow will be drawn from the start location.
    title : str, optional
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new figure and axis are created.
    show_pass : bool, default=True
        If True, an arrow showing the start and end of the pass is included.
    show_surface : bool, default=False
        If True, a heatmap of the prediction surface is included.
    surface_kwargs : dict, optional
        Keyword arguments passed to `imshow` when plotting the surface 
        (e.g., `cmap`, `vmin`, `vmax`, interpolation).
    log : bool, default=False
        If True, the log of the surface values is plotted instead of raw values.

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        If no axis is passed (`ax=None`), the created Matplotlib figure is returned. 
        Otherwise, nothing is returned.

    Notes
    -----
    - Uses `mplsoccer.Pitch` for pitch drawing.
    - Player markers:
        - Blue circles (`o`) for teammates
        - Red crosses (`x`) for opponents
    - If `component='selection'`, teammates' transparency reflects their predicted selection probability.
    - Surface is assumed to be aligned with pitch dimensions (105x68).
    """
    ff_action = freeze_frame['freeze_frame_360_a0']
    if component == 'selection':
        player_vals = results.map_player_to_surface(surface, ff_action)
    ball_x = start['start_x_a0']
    ball_y = start['start_y_a0']
    if ball_speed is not None:
        ball_x_velo = ball_speed['speedx_a02']
        ball_y_velo = ball_speed['speedy_a02']
    if end is not None:
        ball_x_end = end['end_x_a0']
        ball_y_end = end['end_y_a0']
    pitch = mplsoccer.pitch.Pitch(
        pitch_type='custom', 
        half=False,         # Show only half the pitch (positive quadrant)
        pitch_length=105,   # Length of the pitch (in meters)
        pitch_width=68,     # Width of the pitch (in meters)
        goal_type='box',
        axis=True           # Show axis for coordinates
    )
    
    # Create a figure
    show = False
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
        show = True
    else:
        pitch.draw(ax=ax)
    for player in ff_action:
        if player['teammate']:
            color = 'b'
            marker = 'o'
        else:
            color = 'r'
            marker = 'x'
        start_x, start_y = player['x'], player['y']
        end_x, end_y = player['x'] + player['x_velo'], player['y'] + player['y_velo']
        pitch.arrows(start_x, start_y, end_x, end_y, width=1, headwidth=5, color='gray', ax=ax)
        alpha = 1
        if component == 'selection' and player['teammate'] and not player['actor']:
            alpha = player_vals[player['player']]
            vis_alpha = max(alpha, .2)
        else:
            clipped_x = np.clip(start_x / 105 * surface.shape[1], 0, surface.shape[1] - 1).astype(np.uint8)
            clipped_y = np.clip(start_y / 105 * surface.shape[0], 0, surface.shape[0] - 1).astype(np.uint8)
            alpha = surface[int(clipped_y), int(clipped_x)]
            vis_alpha = alpha
            #print(alpha)
        alpha = np.round(alpha, 5)
        vis_alpha = min(vis_alpha, 1)#make sure everything is visible
        if not player['teammate'] or component != 'selection' or player['actor']:
            vis_alpha = 1
        if player['teammate'] and not player['actor']:
            ax.text(start_x, start_y + .5, f'{alpha:.1%}', fontsize=8, color='black', ha='left', va='bottom')
        pitch.scatter([start_x], [start_y], c=color, s=30, ax=ax, marker = marker, alpha = vis_alpha)
    # Scatter the start and end points for clarity
    pitch.scatter([ball_x], [ball_y], c='w', ec = 'k', s=20, ax=ax)
    if ball_speed is not None:
        pitch.arrows(ball_x, ball_y, ball_x + ball_x_velo, ball_y + ball_y_velo, width=1, headwidth=5, color='gray', ax=ax)
    # Set labels
    if show_pass and end is not None:
        pitch.arrows(ball_x, ball_y, ball_x_end, ball_y_end, width = 2, headwidth = 5, color = 'black', ax = ax)
    if show_surface:
        if log:
            ax.imshow(np.log(surface), extent=[0.0, 105.0, 0.0, 68.0], origin='lower', **surface_kwargs)
        else:
            ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin='lower', **surface_kwargs)
    ax.set_title(title)

    if show:
        return fig


def plot_model_outputs(component: str,
                       run_id: str,
                       path_feature: str,
                       path_output: str = 'output/surface.pdf',
                       game_id=None,
                       max_actions: int = 200,
                       **kwargs):
    """
    Generate and save plots of soccer decision-making model outputs.

    Parameters
    ----------
    component : {'selection', 'success', 'value'}
        type of model (response variable).
    run_id : str
        MLflow run ID to load the model from.
    path_feature : str
        Path to feature files to use as input data.
    path_output : str
        Path to save the PDF with plots (default is model_outputs.pdf)
    game_id :
        Optional game ID to limit plotting to one game.
    show_pass : bool, default=True
        If True, an arrow showing the start and end of the pass is included.
    show_surface : bool, default=False
        If True, a heatmap of the prediction surface is included
    max_actions : int, optional
        Maximum number of actions to plot (default is 200).
    **kwargs : dict, optional
        Additional keyword arguments passed directly to visualize_surface.
    """

    # Load the model
    if (component == 'selection') :
        model = pass_selection_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )
    elif (component == 'success') :
        model = pass_success_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )
    elif (component == 'value') :
        model = pass_value_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )

    # Prepare dataset
    features_dir = os.path.join(path_data, path_feature)
    dataset_test = partial(PassesDataset, path=features_dir)

    # Predict surfaces
    surfaces_wSpeed = model.predict_surface(dataset_test, db=None, game_id=game_id)

    # Load freeze frame data
    freeze_frame = pd.read_parquet(os.path.join(features_dir, 'x_freeze_frame_360.parquet'))
    speed = pd.read_parquet(os.path.join(features_dir, 'x_speed.parquet'))
    start = pd.read_parquet(os.path.join(features_dir, 'x_startlocation.parquet'))
    end = pd.read_parquet(os.path.join(features_dir, 'x_endlocation.parquet'))

    surface_type = {"interpolation": "bilinear", "vmin": None, "vmax": None, "cmap": "Greens"}

    # Plot and save to PDF
    count = 0
    with PdfPages(path_output) as pdf:
        for game_id in surfaces_wSpeed:
            for action_id in tqdm(surfaces_wSpeed[game_id]):
                ff_action = freeze_frame.loc[(game_id, action_id)]
                start_action = start.loc[(game_id, action_id)]
                speed_action = speed.loc[(game_id, action_id)]
                end_action = end.loc[(game_id, action_id)]
                surface = surfaces_wSpeed[game_id][action_id]

                title = f"Game {game_id}, index {action_id}, max_val {np.amax(surface)}"
                fig = visualize_surface(
                    component=component,
                    freeze_frame=ff_action,
                    surface=surface,
                    start=start_action,
                    end=end_action,
                    title=title,
                    surface_kwargs=surface_type,
                    log=False,
                    **kwargs,
                )

                pdf.savefig(fig)
                plt.close(fig)

                count += 1
                if count >= max_actions:
                    return



def generate_pass_surface_gifs(
    component: str,
    run_id: str,
    path_feature: str = "Hawkeye/Hawkeye_Features/sequences_tenSecPrior",
    path_play: str = "steffen/sequence_filtered.csv",
    path_output: str = "output/animations",
    num_to_generate: int = 5,
    fps: int = 10,
    dpi: int = 100,
):
    """
    Generate animated GIFs of soccer pass surface predictions.

    Parameters
    ----------
    component : {'selection', 'success', 'value'}
        type of model (response variable).
    run_id : str
        MLflow run ID to load the model from.
    path_feature : str
        Path to feature files to use as input data.
    path_play : str
        Path to CSV file with filtered play sequences.
    path_output : str
        Directory to save generated animations.
    num_to_generate : int
        Number of animations to generate.
    fps : int
        Frames per second for the GIF.
    dpi : int
        Resolution for the GIF.
    """

    os.makedirs(path_output, exist_ok=True)

    # Load the model
    if (component == 'selection') :
        model = pass_selection_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )
    elif (component == 'success') :
        model = pass_success_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )
    elif (component == 'value') :
        model = pass_value_speeds.SoccerMapComponent(
            model=mlflow.pytorch.load_model(f'runs:/{run_id}/model', map_location='cpu')
        )

    # Load the data
    features_dir = os.path.join(path_data, path_feature)
    dataset_test = partial(PassesDataset, path=features_dir)

    frame = (
        model
        .initialize_dataset(dataset=dataset_test)
        .features
        .reset_index()
        .rename(columns={'game_id': 'match_id'})
        .assign(index=lambda d: d['action_id'].str.split('-').str[0].astype(int))
    )

    sequence_to_include = (
        frame
        .loc[:, ['match_id', 'index']]
        .drop_duplicates()
        .assign(include=True)
        .head(num_to_generate)
    )

    subset = np.flatnonzero(
        frame.merge(sequence_to_include, on=['match_id', 'index'], how='left')['include']
    )

    # Pre-compute surfaces
    surfaces = model.predict_surface(dataset_test, subset = subset)

    # Load additional datasets
    freeze_frame_df = pd.read_parquet(f'{path_feature}/x_freeze_frame_360.parquet')
    speed_df = pd.read_parquet(f'{path_feature}/x_speed.parquet')
    start_df = pd.read_parquet(f'{path_feature}/x_startlocation.parquet')

    # Generate animations
    for i, sequence in sequence_to_include.iterrows():
        match_id = sequence['match_id']
        index = sequence['index']
        animation_i = animation.getSurfaceAnimation(
            component=component,
            index=index,
            game_id=match_id,
            surfaces=surfaces,
            freeze_frame=freeze_frame_df,
            start=start_df,
            speed=speed_df,
            log=True,
            title=f"Selection Probabilities | {match_id} | {index}",
            numFrames=251,
            playerOnly=True,
            modelType=component,
        )
        
        animation_title = f"{path_output}/{match_id}_{index}_{component}.gif"
        animation_i.save(animation_title, writer="pillow", fps=fps, dpi=dpi)
