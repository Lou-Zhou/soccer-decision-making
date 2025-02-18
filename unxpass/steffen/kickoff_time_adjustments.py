# step 4a:
# get Statsbomb events, adjust timestamp based on kickoff time

import pandas as pd
from statsbombpy import sb
import json
import warnings
import numpy as np
from statsbombpy.api_client import NoAuthWarning
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=NoAuthWarning, module='statsbombpy')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
def get_kickoff_times(match_id):
    df = sb.events(match_id=match_id)
    kickoff_events = df[
        (df['type'] == 'Pass') & 
        (df['pass_type'] == 'Kick Off')
    ]
    first_kickoffs = kickoff_events.groupby('period').first().reset_index()
    kickoff_times = first_kickoffs[['period', 'timestamp']]
    kickoff_times['match_id'] = match_id
    kickoff_times = kickoff_times.rename(columns={'period': 'half', 'timestamp': 'kickoff_time'})
    return kickoff_times

def adjust_timestamps(df):
    # Convert timestamps and kickoff times to Timedelta
    df['event_time'] = pd.to_timedelta(df['timestamp'])
    df['kickoff_time'] = pd.to_timedelta(df['kickoff_time'])
    
    # Subtract kickoff time from event time
    df['adjusted_time'] = df['event_time'] - df['kickoff_time']
    
    # Set negative times to zero
    df['adjusted_time'] = df['adjusted_time'].apply(lambda x: max(pd.Timedelta(0), x))
    
    # Convert adjusted times back to string format hh:mm:ss.xxx
    df['adjusted_time'] = df['adjusted_time'].apply(lambda x: str(x)[7:] if len(str(x)) > 7 else '00:00:00.000')
    
    # Rename columns: timestamp to timestamp_original, adjusted_time to timestamp
    df = df.rename(columns={'timestamp': 'timestamp_original', 'adjusted_time': 'timestamp'})
    
    return df

def process_multiple_matches(match_ids):
    # Initialize an empty list to store DataFrames for each match
    all_data = []
    
    for match_id in match_ids:
        # Load event data for the match
        df = sb.events(match_id=match_id)
        
        # Get kickoff times for the match
        kickoff_times = get_kickoff_times(match_id)
        
        # Merge kickoff times with the event data
        df = pd.merge(df, kickoff_times, left_on=['match_id', 'period'], right_on=['match_id', 'half'])
        
        # Adjust the timestamps
        df = adjust_timestamps(df)
        
        # Append the adjusted DataFrame to the list
        all_data.append(df)
        #all_data[match_id] = df
    # Concatenate all DataFrames into one
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data
    #return all_data

def process_one_match(input_path, output_dir):
    # Initialize an empty list to store DataFrames for each match
    

    # Load event data for the match
    df = pd.read_json(input_path, convert_dates = False)
    match_id = int(input_path.split("/")[-1].replace(".json", ""))
    print(f"Processing {match_id}")
    df['match_id'] = match_id
    # Get kickoff times for the match
    kickoff_times = get_kickoff_times(match_id)
    
    # Merge kickoff times with the event data
    df = pd.merge(df, kickoff_times, left_on=['match_id', 'period'], right_on=['match_id', 'half'])
    
    # Adjust the timestamps
    df = adjust_timestamps(df)
    df['timestamp'] = np.where(~df['timestamp'].str.contains("\."), df['timestamp'] + ".000", df['timestamp'])
    #print(df['timestamp'])
    # Append the adjusted DataFrame to the list
    #all_data.append(df)
        #all_data[match_id] = df
    # Concatenate all DataFrames into one
    #combined_data = pd.concat(all_data, ignore_index=True)
    #return df
    json_str = df.to_json(orient='records')
    json_data = json.loads(json_str)
    cleaned_data = json_data
    #cleaned_data = remove_nan(json_data)

# Save to a file
    with open(output_dir, 'w') as json_file:
        json.dump(cleaned_data, json_file, indent=2)

# Execution:
#match_ids = [3847567, 3845507, 3844385, 3835338, 3835330, 3835322]
match_ids = [3845507, 3835330, 3835338, 3844385, 3835322, 3847567]
#appended_df = process_multiple_matches(match_ids)

for match_id in match_ids:
#print('timestamps after kickoff adjustment:', appended_df[['match_id', 'period', 'timestamp_original', 'timestamp']].head())
    inputdir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/events/{match_id}.json"
    outputdir = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro_receipts/womens_euro_time_fixed/{match_id}.json"
    process_one_match(inputdir, outputdir)

#print('sb_events:', appended_df)
#appended_df.to_csv('sb_events.csv', index=False)