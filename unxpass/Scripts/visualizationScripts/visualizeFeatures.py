import pandas as pd
import numpy as np
from unxpass.converters import playVisualizers
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from unxpass.databases import SQLiteDatabase
from collections import defaultdict
import json
def main(pdf_filename, sample = 300, random = True):
    dbpath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
    db = SQLiteDatabase(dbpath)
    direct = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_filtered"
    buli_ff= pd.read_parquet(f"{direct}/x_freeze_frame_360.parquet")
    buli_end = pd.read_parquet(f"{direct}/x_endlocation.parquet")
    buli_start = pd.read_parquet(f"{direct}/x_startlocation.parquet")
    buli_speed = pd.read_parquet(f"{direct}/x_speed.parquet")
    distances = pd.merge(buli_start, buli_end, left_index = True, right_index = True)
    distances['distance'] = np.sqrt((distances['end_x_a0'] - distances["start_x_a0"]) ** 2 + (distances['end_y_a0'] - distances["start_y_a0"])**2)
    filtered = distances[distances["distance"] < .5]
    

# Convert to regular dict if desired
    
    if random:
        idxs = buli_ff[~pd.isna(buli_ff["freeze_frame_360_a0"])]
        plotIdxs = idxs.sample(sample).index
    else: 
        plotIdxs = buli_end[:sample].index
    index_dict = defaultdict(list)
    with PdfPages(pdf_filename) as pdf:
        for idx in tqdm(plotIdxs):
            game_id = idx[0]
            play = idx[1]
            fig = playVisualizers.visualize_coords_from_parquet(buli_ff, buli_start, buli_speed, idx, end = buli_end, title = f"{game_id} | {play}")
            pdf.savefig(fig)
            plt.close(fig)
            index_dict[game_id].append(play)
    
    with open('index_dict.json', 'w') as f:
        json.dump(index_dict, f)
if __name__ == "__main__": main("endLocTest.pdf", sample = 300, random = True)
