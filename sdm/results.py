#Generate results for all models - relatively inefficient, might be worth optimizing in the future
import numpy as np
from collections import defaultdict

def map_player_to_surface(surface, players):
    """
    Maps each player to a list of surface values closest to them.
    """
    player_ids = np.array([p['player'] for p in players if p['teammate'] and not p['actor']])
    player_coords = np.array([[p['x'], p['y']] for p in players if p['teammate'] and not p['actor']])

    height, width = surface.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height)) #builds coordinate grids
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2) #combine into a grid of coordinates


    distances = np.linalg.norm(coords[:, None, :] - player_coords[None, :, :], axis=2) 
    #compute all distances, 3d matrix of distances of all players

    #get closest distance
    closest_indices = np.argmin(distances, axis=1)
    closest_players = player_ids[closest_indices] 

    surface_values = surface.flatten()

    player_sums = defaultdict(float)
    for pid, val in zip(closest_players, surface_values):
        player_sums[pid] += val

    return dict(player_sums)
