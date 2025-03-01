import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import re

def convert_Hawkeye(coords, needConvert):
    """
    Convert hawkeye coords to statsbomb coords
    1. Need to convert from meters to yards - done
    2. Flip and resize such that HawkEye coord system goes to statsbomb coord. system
    3. Flip x axis if needed - dependending on GK location - done
    """
    x, y = coords
    x = x * (60 / 52.5) + 60#step 1
    y = y * (40 / 43) + 40#not perfect but this is probably the closest we will get - make sure to change this in og buli code too
    #convert to statsbomb coord system here...
    y = 80 - y#step 2
    if needConvert:
        x = 120 - x
    return [x, y]

def read_Hawkeye_player_loc(file_path, period, minute, second_range, team,actor, action_id, player_to_team, goalkeepers):
    """
    Read hawkeye player locations to statsbomb
    """
    print(second_range)
    player_df_all = pd.read_json(file_path, lines = True, orient = 'columns')
    player_dict = player_df_all['samples'].loc[0]['people']
    player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
    player_df['time'] = player_df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
    player_df = player_df.sort_values(by = ['time'])
    second = second_range[0]
    if len(second_range) == 1:
        max_second = second + 1
    else:
        max_second = second_range[1]
    if second > max_second:
        second = 0
    player_df = player_df[(player_df['time'] >= second) & (player_df['time'] <= max_second)]
    
    print(file_path)
    player_df['minute'] = minute
    player_df['pos'] = player_df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
    player_df['pos_name'] = player_df.apply(lambda d: d['role']['name'], axis = 1)
    player_df = player_df[(player_df['pos_name'] == "Goalkeeper") | (player_df['pos_name'] == "Outfielder")]
    sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
    uefa_map = dict(zip(sequences['uefa_player_id'], sequences['player_id']))
    player_df['statsbombid'] = player_df.apply(lambda d: d['personId']['uefaId'], axis = 1).astype(int).map(uefa_map)
    
    player_df['team'] = player_df['statsbombid'].astype(int).map(player_to_team)#error here - uefa_map might not contain all
    #need full mapping between uefa and statsbomb i think instead of a temporary one built from sequences
    player_df['teammate'] = np.where(player_df['team'] == team, True, False) 
    player_df['isGK'] = np.where(player_df['pos_name'] == "Goalkeeper", True, False)
    player_df['isActor'] = np.where(player_df['statsbombid'].astype(int) == actor, True, False)
    sample_time =  player_df['time'].iloc[0]
    test = player_df[player_df['time'] == sample_time]
    
    #
    player_df['period'] = period
    if player_df[(player_df['isGK']) & (player_df['teammate'])].reset_index(drop = True).loc[0]['pos'][0] > 0:#gotta be a better way to do this...
        player_df['NeedsFlip'] = True
    else:
        player_df['NeedsFlip'] = False
    #convert pos here
    player_df['conv_pos'] = player_df.apply(lambda row: convert_Hawkeye(row['pos'], row['NeedsFlip']), axis=1)
    player_df['freeze_frame'] = player_df.apply(lambda d: {'player': d['personId']['uefaId'],'teammate':d['teammate'], 'keeper':d['isGK'], 'actor':d['isActor'], 
    'location':d['conv_pos']}, axis = 1)
    player_df['frameNum'] = player_df['time'].ne(player_df['time'].shift()).cumsum()
    player_df["event_uuid"]  = action_id + "-" + player_df['frameNum'].astype(str)
    
    threesixty = player_df[['event_uuid','freeze_frame']].reset_index(drop = True).groupby("event_uuid").apply(lambda x: x['freeze_frame'].to_list()).reset_index(name = "freeze_frame")
    threesixty["visible_area"] = [[0,0, 120, 0, 120, 80, 0, 80, 0,0]] * threesixty.shape[0]
    return threesixty[['event_uuid', 'visible_area', 'freeze_frame']]

def convert_hawkeye_to_360(df, idx, goalkeepers, player_to_team):
    """
    Reads hawkeye data and converts to 360 version
    """
    row = df.loc[idx]
    action_id = row['id']
    team = row['team']['id']
    period = row['period']
    time = row['timestamp']
    actor = row['player']['id']
    time_split = time.split(":")
    minute = int(time_split[-2]) + 1
    second = float(time_split[-1])

    player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{str(minute)}.football.samples.centroids"
    #print(player_loc_path)
    player_df = read_Hawkeye_player_loc(player_loc_path, period, minute, [second], team, actor, action_id, player_to_team, goalkeepers)
    all_dfs = [player_df]
    if(second > 59):
        print("Goes to next minute...")
        player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{str(minute + 1)}.football.samples.centroids"
        player_df = read_Hawkeye_player_loc(player_loc_path, period, minute, [0, 60 - second], team,actor, action_id, player_to_team, goalkeepers)
        all_dfs.append(player_df)
    all_player_locs = pd.concat(all_dfs)
    return all_player_locs

def concat_event_data(num_frames, idx, event_df, reverse = False):
    """ 
    Adds 1-second after frames
    """
    row = event_df.loc[idx]
    action_id = row['id']
    df_to_add = pd.DataFrame([row] * num_frames)
    df_to_add['id'] = [f"{action_id}-{frame}" for frame in range(1,num_frames+1)]
    #print(df_to_add.columns)
    new_df = pd.concat([event_df, df_to_add]).sort_values(by = ['period', 'timestamp'])#.reset_index()
    return new_df

def save_to_json(path, df):
    """
    Convert to statsbomb-readable json
    """
    json_str = df.to_json(orient='records')
    json_data = json.loads(json_str)
    #cleaned_data = remove_nan(json_data)

    # Save to a file
    with open(path, 'w') as json_file:
        json.dump(json_str, json_file, indent=2)
    
def add_one_second_frames(match, output_dir):
    """
    Adds one second after each frame and converts to a statsbomb readable json format
    """
    game_id = matches_map[match]
    events_dir = f'/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events/{game_id}.json'
    three_sixty_dir = f'/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/three-sixty/{game_id}.json'
    ball_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye/HawkeyeUnzipped/{match}/{match}ball_loc.json"
    lineups = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/lineups/{game_id}.json"
    #player_tracking_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye/HawkeyeUnzipped/{sample_match}/{sample_match}player_loc.json"
    global player_tracking_dir
    global player_to_team
    global file_path_begin
    global goalkeepers
    global ball_locs
    player_tracking_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye/HawkeyeUnzipped/{match}/scrubbed.samples.centroids/"
    event_output_dir = f"{output_dir}/events_added/{game_id}.json"
    three_sixty_output = f"{output_dir}/three-sixty_added/{game_id}.json"
    events_df = pd.read_json(events_dir, convert_dates = False)
    three_sixty_df = pd.read_json(three_sixty_dir, convert_dates = False)
    ball_locs = pd.read_json(ball_dir, convert_dates = False)
    lineup_df = pd.read_json(lineups, convert_dates = False)
    dirfiles = [f for f in listdir(player_tracking_dir) if not f.startswith('.')]
    sample_file = dirfiles[0]
    file_path_begin = "_".join(sample_file.split('_')[0:3])
    #player_locs = pd.read_json(player_tracking_dir, convert_dates = False)
    #player_locs['added_time'] = player_locs['added_time'].fillna(0)
    ball_locs['added_time'] = ball_locs['added_time'].fillna(0)
    team_1 = lineup_df['team_id'].loc[0]
    team_2 = lineup_df['team_id'].loc[1]
    team_1_dict = lineup_df['lineup'].loc[0]
    team_2_dict = lineup_df['lineup'].loc[1]
    team_1_lineup = [player_dict['player_id'] for player_dict in team_1_dict]
    team_2_lineup = [player_dict['player_id'] for player_dict in team_2_dict]
    team_map = {team_1 : team_1_lineup, team_2 : team_2_lineup}
    player_to_team = {player_id: team_id for team_id, players in team_map.items() for player_id in players}
    pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_1_dict if len(player['positions']) > 0}
    team_2_pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_2_dict if len(player['positions']) > 0}
    pos_dict.update(team_2_pos_dict)
    goalkeepers = [key for (key,value) in pos_dict.items() if value == "Goalkeeper"]
    needs_converting = events_df[events_df['pass'].notna()].index
    counter = 1
    for idx in needs_converting:
        print(f"\r{counter}/{needs_converting.shape[0]}",end = "", flush=True)
        counter = counter + 1
        new_360 = convert_hawkeye_to_360(events_df, idx, player_to_team, goalkeepers)
        events_df = concat_event_data(new_360.shape[0], idx, events_df)
        three_sixty_df = pd.concat([three_sixty_df, new_360])
    save_to_json(events_df, event_output_dir)
    save_to_json(three_sixty_df, three_sixty_output)
    
def get_receipts(match_id, json_path, output_path):
    """
    Converts statsbomb receipts to passes
    """
    match_path = f"{json_path}{match}.json"
    output_path = f"{output_path}{match}.json"
    test_json = pd.read_json(match_url)
    #can we just do an empty pass - see what happens
    test_json['type'] = np.where(test_json["type"] == {'id': 30, 'name': 'Pass'}, {'id': -1, 'name': 'Old_Pass'}, test_json['type'])
    test_json['type'] = np.where(test_json["type"] == {'id': 42, 'name': 'Ball Receipt*'},{'id': 30, 'name': 'Pass'}, test_json['type'])
    json_str = test_json.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
        #cleaned_data = remove_nan(json_data)
    print(output_path)
    # Save to a file
    with open(output_path, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=2)

def convert_ball_locs(dir_path, output_path):
    """
    Mass convert hawkeye ball locations to statsbomb format
    """
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
        if len(file_split) == 8:#added time
            added = file_split[5]
        ball_df = pd.read_json(file_path, lines=True, orient="columns")
        ball_dict = ball_df["samples"].loc[0]['ball']
        full_df = pd.DataFrame(ball_dict)
        full_df['minute'] = minute
        full_df['added_time'] = added
        full_df['half'] = half
        print(minute, added, half)
        output_dfs.append(full_df)
    all_data = pd.concat(output_dfs).sort_values(by = ['minute', 'time']).reset_index(drop = True)
    json_str = all_data.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
    with open(output_path, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=2)
    #all_data.to_json(output_path, orient='records', lines=True)
    return all_data
def convert_player_locs(dir_path, output_path):
    """
    Mass Convert Hawkeye player locations to statsbomb format
    """
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
        if len(file_split) == 8:#added time
            added = file_split[5]
        #test['details'].loc[0]
        player_df['time'] = player_df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
        player_df['pos'] = player_df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
        player_df['minute'] = minute
        player_df['added_time'] = added
        player_df['half'] = half
    output_dfs.append(player_df)
    all_data = pd.concat(output_dfs).sort_values(by = ['minute', 'time']).reset_index(drop = True)
    json_str = all_data.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
    #with open(output_path, 'w') as json_file:
    #    json.dump(cleaned_data, json_file, indent=2)
    #all_data.to_json(output_path, orient='records', lines=True)
    return all_data
