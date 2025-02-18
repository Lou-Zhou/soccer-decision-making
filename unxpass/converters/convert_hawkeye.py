#Main script to convert from hawkeye to statsbomb
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import re
def convert_ball_locs(dir_path, output_path):
    output_dfs = []
    dirfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for file in dirfiles:
        if(file.startswith('.')):
            print(f"Skipping Hidden File {file}")
            continue#this is stupid but whatever
        file_path = dir_path + file
        print(f"Parsing {file}")
        file_split = re.split(r'[_\.]', file)
        half = file_split[3]
        minute = file_split[4]
        added = None
        if len(file_split) == 9:#added time
            added = file_split[5]
        ball_df = pd.read_json(file_path, lines=True, orient="columns")
        ball_dict = ball_df["samples"].loc[0]['ball']
        full_df = pd.DataFrame(ball_dict)
        full_df['minute'] = minute
        full_df['added_time'] = added
        full_df['half'] = half
        #print(minute, added, half)
        output_dfs.append(full_df)
    print("Finished Parsing...")
    print(f"Putting together {len(output_dfs)} dataframes")
    all_data = pd.concat(output_dfs).sort_values(by = ['minute', 'time']).reset_index(drop = True)
    json_str = all_data.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
    with open(output_path, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=2)
    #all_data.to_json(output_path, orient='records', lines=True)
    return all_data
def convert_player_locs(dir_path, output_path):#this is too big
    output_dfs = []
    dirfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for file in dirfiles:
        if(file.startswith('.')):
            print(f"Skipping Hidden File {file}")
            continue#this is stupid but whatever
        print(f"Parsing {file}")
        file_path = dir_path + file
        player_df_all = pd.read_json(file_path, lines = True, orient = 'columns')

        player_dict = player_df_all['samples'].loc[0]['people']
        player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
        file_split = re.split(r'[_\.]', file)
        half = file_split[3]
        minute = file_split[4]
        added = None
        if len(file_split) == 9:#added time
            added = file_split[5]
        #test['details'].loc[0]
        player_df['time'] = player_df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
        player_df['pos'] = player_df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
        player_df['minute'] = minute
        player_df['added_time'] = added
        player_df['half'] = half
        #
        #print(minute, added, half)
        #print(player_df['minute'])
        output_dfs.append(player_df)
    all_data = pd.concat(output_dfs).sort_values(by = ['minute', 'time']).reset_index(drop = True)
    print("Finished Parsing...")
    json_str = all_data.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
    with open(output_path, 'w') as json_file:
        print("Writing Json...")
        json.dump(cleaned_data, json_file, indent=2)
    #all_data.to_json(output_path, orient='records', lines=True)
    return all_data
og_filepath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkEye"
dirfiles = [f for f in listdir(og_filepath)]
print(dirfiles)
for game in dirfiles:
    if(game.startswith('.')):
        print(f"Skipping {game}")
        continue#this is stupid but whatever
    print(f"Parsing {game}")
    outputdir = og_filepath + "/" + game
    subdirfiles = [f for f in listdir(outputdir)]
    player_loc_output = f"{outputdir}/{game}player_loc.json"
    ball_loc_output = f"{outputdir}/{game}ball_loc.json"
    player_loc_input = f"{outputdir}/scrubbed.samples.centroids/"
    ball_loc_input = f"{outputdir}/scrubbed.samples.ball/"
    convert_ball_locs(ball_loc_input, ball_loc_output)
    #convert_player_locs(player_loc_input, player_loc_output)