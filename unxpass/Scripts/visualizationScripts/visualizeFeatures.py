#visualizes from parquet
import pandas as pd
import numpy as np
from unxpass.visualizers import plotPlays
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from unxpass.databases import SQLiteDatabase
from collections import defaultdict
import json
def main(pdf_filename, sample = 300, random = True):
    direct = "../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_test"
    recept = "../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_filtered"
    buli_ff= pd.read_parquet(f"{direct}/x_freeze_frame_360.parquet")
    buli_end = pd.read_parquet(f"{direct}/x_endlocation.parquet")
    buli_start = pd.read_parquet(f"{direct}/x_startlocation.parquet")
    buli_speed = pd.read_parquet(f"{direct}/x_speed.parquet")
    buli_success = pd.read_parquet(f"{direct}/y_success.parquet")
    recipient_end = pd.read_parquet(f"{recept}/x_endlocation.parquet")
    distances = pd.merge(buli_start, buli_end, left_index = True, right_index = True)
    distances = pd.merge(distances, buli_success, left_index = True, right_index = True)
    distances['distance'] = np.sqrt((distances['end_x_a0'] - distances["start_x_a0"]) ** 2 + (distances['end_y_a0'] - distances["start_y_a0"])**2)
    filtered = distances[(distances["distance"] > 40) & (~distances['success']) & ((distances['end_y_a0'] < 5) | (distances['end_y_a0'] > 60))]
    print(f"Successful {filtered['success'].sum()} / {len(filtered)}")
    recept_success = pd.read_parquet(f"{recept}/y_success.parquet")
    recept_success = recept_success[~recept_success['success']]
# Convert to regular dict if desired
    
    if random:
        plotIdxs = recept_success.sample(sample).index
    else: 
        plotIdxs = recept_success[:sample].index
    index_dict = defaultdict(list)
    with PdfPages(pdf_filename) as pdf:
        for idx in tqdm(plotIdxs):
            game_id = idx[0]
            play = idx[1]
            successful = buli_success.loc[idx]['success']
            ff = buli_ff.loc[idx]
            start = buli_start.loc[idx]
            speed = buli_speed.loc[idx]
            end = buli_end.loc[idx]
            recept_end = recipient_end.loc[idx]
            fig = visualize_coords_from_parquet(ff, start, speed, idx, title = f"Game {game_id} | Action {play}", surface = None, surface_kwargs = None, ax = None, log = False, playerOnly = False, modelType = "sel")
            pdf.savefig(fig)
            plt.close(fig)
            index_dict[game_id].append(play)
    
    with open('index_dict.json', 'w') as f:
        json.dump(index_dict, f)
if __name__ == "__main__": main("weirdTest.pdf", sample = 300, random = True)
