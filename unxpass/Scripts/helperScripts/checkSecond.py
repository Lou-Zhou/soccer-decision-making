#ensure that all hawkeye events of interest have atleast one second to evaluate
#if not, then need to trim to not include irrelevant game states
from unxpass import load_xml
import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
def trimSecondIdx(sequences, hawkeye_events, seconds_after = 1):
    allGames = []
    for game in tqdm(hawkeye_events):
        hawkeyeEvents = pd.read_json(f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/events/{game}")[['id', 'timestamp', 'period', 'index']]
        hawkeyeEvents = hawkeyeEvents.sort_values(by = ['period', 'timestamp'])
        hawkeyeEvents['match_id'] = game.split(".")[0]
        m = hawkeyeEvents['timestamp'].ne(hawkeyeEvents['timestamp'].shift())
        # mask the other, shift up, backfill
        hawkeyeEvents['next_diff_time'] = hawkeyeEvents['timestamp'].where(m).shift(-1).bfill()
        hawkeyeEvents['time_diff'] = (hawkeyeEvents['next_diff_time'] - hawkeyeEvents['timestamp']).dt.total_seconds()
        hawkeyeEvents = hawkeyeEvents[hawkeyeEvents['id'].isin(sequences['id'])]
        allGames.append(hawkeyeEvents)
    allGames = pd.concat(allGames)
    print(f"Maximum Time After {max(allGames['time_diff'])}")
    allGames = allGames[allGames['time_diff'] > seconds_after]
    print(f"Number within {seconds_after} second: {len(allGames)} out of {len(sequences)}")
    allGames['numFrames'] = np.rint(allGames['time_diff'] / .04)
    return allGames
    excludeIdx = []
    for idx, row in allGames.iterrows():
        for i in range(25, int(row['numFrames'] - 1), -1):
            index = (int(row['match_id']), f"{row['index']}-{i}")
            excludeIdx.append(index)
    return excludeIdx
def main():
    sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequence_filtered.csv", delimiter = ";")
    #hawkeyeEvents
    hawkeye_events = os.listdir("/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/events")
    excludeIdx = trimSecondIdx(sequences, hawkeye_events, seconds_after = 3)
    #print(excludeIdx)
if __name__ == "__main__": main()