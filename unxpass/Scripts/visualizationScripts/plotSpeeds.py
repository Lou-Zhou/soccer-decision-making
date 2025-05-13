#visualizes speed of a player in a game
import pandas as pd
import numpy as np
import os
import regex as re
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from unxpass import load_xml

pd.options.mode.chained_assignment = None

def savetopdf(pdf_filename, allpos, framesback):
    idx = 1
    with PdfPages(pdf_filename) as pdf:
        for player in allpos['uefaID'].unique():
            print(f"Iterating player {player}, {idx}/{len(allpos['uefaID'].unique())}", end = "\r",)
            idx += 1
            playerpos = allpos[allpos['uefaID'] == player]
            fig = generateSpeedPlots(playerpos, framesback=framesback)
            pdf.savefig(fig, bbox_inches="tight")  # Save figure to PDF
            plt.close(fig) 
#HAWKEYE SPEED
def json_player_to_df(filepath):
    player_df_all = pd.read_json(filepath, lines = True, orient = 'columns')
    player_dict = player_df_all['samples'].loc[0]['people']
    player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
    player_df['time'] = player_df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
    player_df['uefaID'] = player_df.apply(lambda d: d['personId']['uefaId'], axis = 1)
    player_df['position'] = player_df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
    player_df['role'] = player_df.apply(lambda d: d['role']['name'], axis = 1)
    player_df = player_df[(player_df['role'] == "Goalkeeper") | (player_df['role'] == "Outfielder")]
    filename = filepath.split("/")[-1]
    filesplit = re.split("\_|\.",filename)
    period = filesplit[3]
    minute = filesplit[4]
    if len(filesplit) == 8:
        added_time = 0
    else:
        added_time = filesplit[5]
    #player_df = player_df.sort_values(by = ['time'])
    player_df["period"] = int(period)
    player_df['minute'] = int(minute)
    player_df['added_time'] = int(added_time)
    return player_df[['time', 'uefaID','position', "period", "minute", "added_time", 'role']]

def generateSpeedPlots(playerpos, framesback):
    first_min = {2:46, 3:91, 4: 106}
    playerpos['prev_pos'] = playerpos['position'].shift(framesback)
    playerpos['prev_pos'] = playerpos['prev_pos'].fillna(playerpos['position'])
    playerpos['current_fulltime'] = playerpos['time'] + 60 * (playerpos['minute'] + playerpos['added_time'])
    playerpos['prev_time'] = playerpos['current_fulltime'].shift(framesback)
    playerpos['speed'] = playerpos.apply(lambda row: distance.euclidean(
        row['position'], row['prev_pos']) / (row['current_fulltime'] - row['prev_time']), axis = 1)
    playerpos['speed'] = playerpos['speed'].fillna(0)
    for period in range(1, max(playerpos['period'])):
        minute = first_min[period + 1]
        time = 0
        mask = (playerpos['time'] == time) & (playerpos['minute'] == minute) & (playerpos['period'] == period + 1)
        playerpos.loc[mask, "speed"] = 0
    playerpos = playerpos.reset_index(drop = True)
    fig, axs = plt.subplots(2,1)
    fig.suptitle(f"Speed Plots, {framesback} frames back, player: {playerpos['uefaID'].iloc[0]}, position: {playerpos['role'].iloc[0]}")
    sns.lineplot(data=playerpos, x="frame", y="speed", ax=axs[0])
    axs[0].set(title = "Speed(m/s) vs All Frames")
    sns.lineplot(data=playerpos[playerpos['minute'] == 40], x="frame", y="speed", ax=axs[1])
    axs[1].set(title = "Speed(m/s) vs 40th Minute Frames")
    
    plt.tight_layout()
    return fig
def getHawkeyeSpeedPlots(base_filepath, framesback, outputpdf):
    print("Plotting Hawkeye Speeds...")
    player_filepath = f"{base_filepath}/scrubbed.samples.centroids"
    all_files = []
    iter = 1
    for filename in os.listdir(player_filepath):
        print(f"Processing {filename}, {iter}/{len(os.listdir(player_filepath))}", end = "\r")
        testfile = f"{player_filepath}/{filename}"
        player_df = json_player_to_df(testfile)
        all_files.append(player_df)
        iter += 1
    #case of one player
    allpos = pd.concat(all_files)
    pdf_filename = outputpdf
    allpos = allpos.sort_values(by = ["period", "minute", "added_time", "time"])
    allpos['frame'] = allpos.groupby(['period', 'minute', 'added_time', 'time']).ngroup()
    idx = 1
    savetopdf(outputpdf, allpos, framesback)


#BULI SPEED
from unxpass import load_xml
def plotBuLiSpeeds(game_id, framesback, outputpdf):
    print(f"Plotting {game_id}")
    trackingdf = load_xml.load_tracking(f"../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/{game_id}.xml")
    trackingdf['frame'] = trackingdf['N'].astype(int)
    trackingdf['X'] = trackingdf['X'].astype(float)
    trackingdf['Y'] = trackingdf['X'].astype(float)
    trackingdf = trackingdf.sort_values(by = "N")
    inithalfs = trackingdf.drop_duplicates('GameSection')[['GameSection', "T"]]
    inithalfs['period'] = range(1, inithalfs.shape[0] + 1)
    trackingdf = pd.merge(trackingdf, inithalfs, on = "GameSection")
    half_dict = {1:0, 2:45, 3:90, 4:105}
    trackingdf['totaltime'] = (pd.to_datetime(trackingdf['T_x']) - pd.to_datetime(trackingdf['T_y'])).dt.total_seconds()
    trackingdf['minute'] = trackingdf['period'].map(half_dict) + trackingdf['totaltime'] // 60 + 1
    trackingdf['time'] = trackingdf['totaltime'] % 60 #this is kinda dumb but ig its better to reuse old functions
    trackingdf['added_time'] = 0#just a dummy
    trackingdf['position'] = trackingdf[['X','Y']].values.tolist()
    trackingdf['uefaID'] = trackingdf['PersonId']#also stupid but whatevs, not actual uefaID
    trackingdf['role'] = ""
    trackingdf_clean = trackingdf[["uefaID", "frame", "position", "period", "totaltime", "minute", "time", "added_time", "role"]]
    savetopdf(outputpdf, trackingdf_clean, framesback)


base_filepath = "../../../../rdf/sp161/shared/soccer-decision-making/allHawkeye/2032206_England_Austria"
getHawkeyeSpeedPlots(base_filepath, 1, "HawkeyeSpeedPlots1Frame.pdf")
getHawkeyeSpeedPlots(base_filepath, 10, "HawkeyeSpeedPlots10Frame.pdf")


game_id = "DFL-MAT-J03YDU"
plotBuLiSpeeds(game_id, 1, "BuLiSpeedPlots1Frame.pdf")
plotBuLiSpeeds(game_id, 10, "BuLiSpeedPlots10Frame.pdf")