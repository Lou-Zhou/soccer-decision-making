#Script which adds a second after all data in sequences
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import json
game_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkEye/"#directory of hawkeye data
games = pd.read_json("/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/matches/53/106.json", convert_dates = False)#directory of statsbomb matches data
games['home_team'] = games.apply(lambda d: d['home_team']['home_team_name'], axis = 1).str.replace("Women's", "").str.replace("WNT", "").str.strip()
games['away_team'] = games.apply(lambda d: d['away_team']['away_team_name'], axis = 1).str.replace("Women's", "").str.replace("WNT", "").str.strip()
#dirfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
dirfiles = [f for f in listdir(game_dir) if not f.startswith('.')]
matches_map = {}
missinggks = []
period_adjustment = {1:0, 2:45, 3:90, 4:105}
time_dict ={}
sequences = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/sequences_new.csv")
for index, row in sequences.iterrows():
    if(row['BallReceipt'] != row["BallReceipt"]):
        continue
    minutes = int(row['BallReceipt'] // 60) + 1
    seconds = row['BallReceipt'] % 60
    time_dict[row['id']] = [minutes, seconds]
for game in dirfiles:
    home_team = game.split('_')[1]
    away_team = game.split('_')[2]
    #print(games['away_team'])
    game_id = games[(games['home_team'] == home_team) & (games['away_team'] == away_team)].reset_index().loc[0]['match_id']
    matches_map[game] = game_id

# %%

# %%
def convert_pos(coords, needConvert):
    """
    Convert to statsbomb coords
    1. Need to convert from meters to yards - done
    2. Flip and resize such that HawkEye coord system goes to statsbomb coord. system
    3. Flip x axis if needed - dependending on GK location - done
    Revisit this - slightly offset
    """
    #try to find linear function between new_x ~ old_x, needsFlip and new_y ~ old_y, needsFlip
    x, y = coords
    x = x * (60 / 52.5) + 60#step 1
    y = y * (40 / 34) + 40#not perfect but this is probably the closest we will get - make sure to change this in og buli code too
    #convert to statsbomb coord system here...
    if needConvert:
        x = 120 - x
    return [x, y]
    #if in right half, need to flip ys?
def get_ball_pos(event_df, idx, raw_df):
    row = event_df.loc[idx]
    period = row['period']
    time = row['timestamp']
    action_id = row['id']
    time_split = time.split(":")
    
    times = time_dict[action_id]
    #minute = int(time_split[-2]) + period_adjustment[period] + 1
    minute = times[0] + period_adjustment[period]
    first_time = raw_df['time'].reset_index(drop = True).loc[0]
    #print(action_id, first_time)
    
    test_loc = ball_locs[(ball_locs['minute'] == minute) & (ball_locs["half"] == period)]
    #print(period, minute)
    #print(ball_locs[ball_locs['half'] == period]['minute'])
    if float(first_time) in set(test_loc['time']):
        ball_location = ball_locs[(ball_locs['minute'] == minute) & (ball_locs["half"] == period) & (ball_locs['time'] == first_time)].reset_index(drop = True).loc[0]["pos"]
    else:
        print("Not Found, finding closest")#ball and player data aren't perfectly synced?
        #print(test_loc.iloc[(test_loc['time']-first_time).abs().argsort()].reset_index(drop = True))
        df_sort = test_loc.iloc[(test_loc['time']-first_time).abs().argsort()].reset_index(drop = True).loc[0]
        new_time = df_sort['time']
        print(new_time, first_time)
        ball_location = ball_locs[(ball_locs['minute'] == minute) & (ball_locs["half"] == period) & (ball_locs['time'] == new_time)].reset_index(drop = True).loc[0]["pos"]
    ball_location = ball_location[0:2]
    return convert_pos(ball_location, raw_df['NeedsFlip'].reset_index(drop=True).loc[0])
# %%
def read_player_loc(file_path, period, minute, second, team,actor, action_id, max_second = None):
    """
    Converts raw Hawkeye data to statsbomb-360 structure, calculating in the interval [second, max_second]
    file_path - string of file of the location of players at the appropriate minute
    minute, second - the minute and second of the reception(second also contains milliseconds(e.g. #.####))
    team - statsbomb id of acting team
    actor - statsbomb id of actor
    action_id - the statsbomb id of the event
    max_second - the upper bound of the second to calculate
    """
    #print(file_path)
    print(file_path)
    player_df_all = pd.read_json(file_path, lines = True, orient = 'columns')
    player_dict = player_df_all['samples'].loc[0]['people']
    player_df = pd.DataFrame(player_dict)#['centroid'].loc[0][0]
    player_df['time'] = player_df.apply(lambda d: d['centroid'][0]['time'], axis = 1)
    #print(player_df)
    player_df = player_df.sort_values(by = ['time'])
    if not max_second:
        max_second = second + 1
    #print(player_df['time'])
    player_df = player_df[(player_df['time'] > second) & (player_df['time'] < max_second)]
    print(action_id)
    print(minute)
    print(second)
    print(player_df['time'])
    if(player_df.shape[0] == 0):
        print("No Overflow")#large second difference?
        return [-1, -1]
    player_df['period'] = period
    player_df['minute'] = minute
    player_df['pos'] = player_df.apply(lambda d: d['centroid'][0]['pos'], axis = 1)
    old = player_df.shape[0]
    player_df = player_df[player_df['personId'].map(lambda x: 'uefaId' in x)]#filter all empty
    new = player_df.shape[0]
    if(old != new):
        print(f"Empty UefaIDs, difference of {old - new}")
    player_df['statsbombid'] = player_df.apply(lambda d: d['personId']['uefaId'], axis = 1).astype(int).map(uefa_map)
    player_df = player_df.dropna()
    player_df['team'] = player_df['statsbombid'].astype(int).map(player_to_team)
    player_df['teammate'] = np.where(player_df['team'] == team, True, False) 
    player_df['isGK'] = np.where(player_df['statsbombid'].isin(goalkeepers), True, False)#finds location of teammates goalkeeper
    player_df['isActor'] = np.where(player_df['statsbombid'].astype(int) == actor, True, False)#this isn't being rendered for some reason sometimes
    player_df["NeedsFlip"] = False
    if player_df[(player_df['isGK']) & (player_df['teammate'])].shape[0] == 0:
        print("No GK Found")
        missinggks.append([action_id])  
    elif player_df[(player_df['isGK']) & (player_df['teammate'])].reset_index(drop = True).loc[0]['pos'][0] > 0:
        player_df['NeedsFlip'] = True
    #convert pos here
    def generate_freeze_frame(group):
        return group.apply(lambda row: {
            "teammate": row["teammate"],
            "keeper": row["isGK"],
            'actor':row['isActor'],
            'location':row['conv_pos']
        }, axis=1).tolist()
    player_df['conv_pos'] = player_df.apply(lambda row: convert_pos(row['pos'], row['NeedsFlip']), axis=1)
    player_df['freeze_frame'] = player_df.apply(lambda d: {'teammate':d['teammate'], 'keeper':d['isGK'], 'actor':d['isActor'], 
    'location':d['conv_pos']}, axis = 1)
    player_df['frameNum'] = player_df['time'].ne(player_df['time'].shift()).cumsum()
    grouped_df = player_df.groupby("frameNum").apply(lambda x: pd.Series({"freeze_frame": generate_freeze_frame(x)})).reset_index()
    grouped_df["event_uuid"]  = action_id + "-" + grouped_df['frameNum'].astype(str)
    grouped_df["visible_area"] = [[0,0, 120, 0, 120, 80, 0, 80, 0,0]] * grouped_df.shape[0]
    
    return grouped_df[['event_uuid', 'visible_area', 'freeze_frame']].reset_index(drop = True), player_df
    #return player_df#.dropna()

# %%
def convert_to_360(df, idx):
    row = df.loc[idx]
    action_id = row['id']
    team = row['team']['id']
    period = row['period']
    time = row['timestamp']
    actor = row['player']['id']
    time_split = time.split(":")
    times = time_dict[action_id]
    print(f"og_time: {time}")
    minute = times[0]
    second = times[1]
    minute = minute + period_adjustment[period]
    #minute = int(time_split[-2]) + 1
    #second = float(time_split[-1])
    #print(minute)
    #print(period, minute,second )
    max_min ={1:45, 2:90, 3:105, 4:120}
    if (minute > max_min[period]):
        player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{max_min[period]}_{minute - max_min[period]}.football.samples.centroids"
    else:
        player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{str(minute)}.football.samples.centroids"
    #print(player_loc_path)
    player_df, raw_player_df = read_player_loc(player_loc_path, period, minute, second, team, actor, action_id)
    all_dfs = [player_df]
    if(second > 59):
        print("Goes to next minute...")
        #(period == 1 & minute > 45) or (period == 2 & minute > 90)
        if (period == 1 and minute + 1 > 45) or (period == 2 and minute + 1 > 90) or (period == 3 and minute + 1 > 105) or (period == 4 and minute + 1 > 120):
            player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{str(max_min[period])}_{minute - max_min[period] + 1}.football.samples.centroids"
        else:
            player_loc_path = f"{player_tracking_dir}{file_path_begin}_{str(period)}_{str(minute + 1)}.football.samples.centroids"
        #print(player_loc_path)
        player_df = read_player_loc(player_loc_path, period, minute, 0, team,actor, action_id, 60 - second)[0]
        #print(player_df)
        if isinstance(player_df, pd.DataFrame):
            all_dfs.append(player_df)
    all_player_locs = pd.concat(all_dfs)
    return all_player_locs, raw_player_df

# %%
#Next TODO: load event data
"""
General thought process:
1. Get number of frames(number of "passes" to add for each event) - call this number n
2. Right below the pass receipt, add n identical passes with the same values except the id will be og_id-k where k is the frame number
Will time be a problem? maybe problems with time
can use .loc[n.1] and then reset_index() maybe?
maybe use timestamp? concat and then sort by timestamp and period
"""
#events_df
def concat_event_data(num_frames, idx, event_df, ball_loc):
    row = event_df.loc[idx]
    action_id = row['id']
    #print(action_id)
    df_to_add = pd.DataFrame([row] * num_frames)
    df_to_add['id'] = [f"{action_id}-{frame}" for frame in range(1,num_frames+1)]
    df_to_add['location'] = [ball_loc] * df_to_add.shape[0]
    #print(df_to_add['location'])
    #print(df_to_add.columns)
    new_df = pd.concat([event_df, df_to_add]).sort_values(by = ['period', 'timestamp'])#.reset_index()
    #print(df_to_add['pass'])
    return new_df

# %%
def save_to_json(df, path):
    json_str = df.to_json(orient='records')
    json_data = json.loads(json_str)
    #cleaned_data = remove_nan(json_data)

    # Save to a file
    with open(path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

# %%
def get_frames(match, output_dir):
    game_id = matches_map[match]
    events_dir = f'/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events_receipt/{game_id}.json'
    three_sixty_dir = f'/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/three-sixty/{game_id}.json'
    ball_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkEye/{match}/{match}ball_loc.json"
    lineups = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Hawkeye_AllGames/lineups/{game_id}.json"
    #player_tracking_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/HawkEye/HawkeyeUnzipped/{sample_match}/{sample_match}player_loc.json"
    global player_tracking_dir
    global player_to_team
    global file_path_begin
    global goalkeepers
    global ball_locs
    player_tracking_dir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/allHawkEye/{match}/scrubbed.samples.centroids/"
    event_output_dir = f"{output_dir}/events/{game_id}.json"
    three_sixty_output = f"{output_dir}/three-sixty/{game_id}.json"
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
    ball_locs['minute'] = ball_locs['minute'] + ball_locs['added_time']
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
    #print(events_df['id'])
    #events_df = events_df[events_df['id'].astype(str).isin(sequences['id'])]
    needs_converting = events_df[(events_df['pass'].notna()) & (events_df['id'].astype(str).isin(sequences['id']))].index

    #print(needs_converting)
    counter = 1
    for idx in needs_converting:
        row = events_df.loc[idx]
        if row['id'] not in time_dict:
            print("not found - no timestamp")#if timestamp is empty
            continue
        #print(row['id'])
        print(f"\r{counter}/{needs_converting.shape[0]}",end = "", flush=True)
        #print(f"\n{idx}")
        counter = counter + 1
        new_360, raw_df = convert_to_360(events_df, idx)
        og_ball_pos = get_ball_pos(events_df, idx, raw_df)
        events_df = concat_event_data(new_360.shape[0], idx, events_df, og_ball_pos)
        three_sixty_df = pd.concat([three_sixty_df, new_360])
    save_to_json(events_df, event_output_dir)
    save_to_json(three_sixty_df, three_sixty_output)
    
    
    


# %%
#3835332
output_dir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/hawkeye_one_game"
#uefa_map = pd.read_csv("/home/lz80/un-xPass/unxpass/steffen/ID_matched.csv").set_index('ID').to_dict()['player_id']
uefa_map = dict(zip(sequences['uefa_player_id'], sequences['player_id']))
reverse_map = dict(zip(sequences['player_id'], sequences['uefa_player_id']))
inv_map = {v: k for k, v in matches_map.items()}
test_idx = dirfiles.index("2032218_Netherlands_Sweden")
match_counter = 1
for match in dirfiles[2:3]:
    match = inv_map[3835331]
    #match = "2032218_Netherlands_Sweden"
    #match = "2032216_Finland_Germany"
    #match = "2032233_France_Netherlands"
    #match = inv_map[3835332]
    print(f"{match} | {matches_map[match]} | {match_counter} / {len(dirfiles)}\n")
    match_counter = match_counter + 1
    get_frames(match, output_dir)#not appending the right things - check this!
print("Finished")
print(missinggks)


"""Missing: [['99a50f47-81f4-435f-96cd-352823631960'], ['02a703cf-94e8-409a-8bd1-1e90c9ff40bc'],
 ['8f3d07fb-9713-4505-bd91-57d309d2db3a'], ['07f9a3eb-4a64-4a91-929f-282eeeb08d63'], ['9c3adb70-8382-4616-83fb-7a0f1f555a76'], 
 ['b5a76709-0482-47d1-a8cd-f94ae97d103a'], ['1ee46fe0-09aa-4be2-9058-b9704087edd9'], ['347cf5ff-31fd-44f6-a92a-17dc6517aa1c'],
  ['6530567b-e92e-4371-bdbf-9c8436ed394e'], ['262076d9-c714-48c8-a8b6-da0f8f706e0b'], ['e9306c1f-4d39-40e8-a97d-7f98a5681a77']]
"""