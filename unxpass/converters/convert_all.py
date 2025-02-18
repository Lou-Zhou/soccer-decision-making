#Main script to convert from buli to statsbomb
import signal
from unxpass.convert_lineups_gamedata import convert_gamedata_lineups
from unxpass.converttrackingevent import convert_event_and_tracking
from os import listdir
from os.path import isfile, join
errors = {}
# Custom exception for timeout
class TimeoutException(Exception):
    pass

# Handler function to raise a timeout exception
def timeout_handler(signum, frame):
    raise TimeoutException("Processing took too long!")

# Define a timeout for the whole processing block
def process_file_with_timeout(file, timeout_seconds=10):
    filename = file.split(".")[0]
    print(f"Processing {filename}...")

    # Construct paths
    players = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/match_information/{filename}.xml"
    tracking = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/{filename}.xml"
    together_csv = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/KPI_Merged_all/KPI_MGD_{filename}.csv"
    event = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/event_data_all/{filename}.xml"
    matchplan = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/match_plan/matchplan.xml"
    gamedata = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/match_plan/matches_DFL-SEA-0001K7.csv"
    event_outpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/events/{filename}.json"#converted_data
    tracking_outpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/three-sixty/{filename}.json"
    lineups_outpath = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/lineups/{filename}.json"
    matches = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/matches/9/281.json"

    # Set the signal handler and timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)  # Sete timeout for the try block
    #convert_event_and_tracking(filename, players, tracking, together_csv, event, matchplan, event_outpath, tracking_outpath)
    try:
        # Call conversion functions
        convert_event_and_tracking(filename, players, tracking, together_csv, event, matchplan, event_outpath, tracking_outpath)
        convert_gamedata_lineups(gamedata, players, lineups_outpath, matches)
        print(f"Processing for {filename} completed successfully.")
    
    except TimeoutException as e:
        errors[filename] = "TimeOut"
        print(f"Timeout while processing {filename}: {e}")
    
    except Exception as e:
        errors[filename] = str(e)
        print(f"An error occurred while processing {filename}: {e}")
    
    finally:
        # Reset the alarm
        signal.alarm(0)

# Main script execution
mypath = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))
#print(len(onlyfiles))
import os
for file in onlyfiles:
    print(file)
    filename = file.split(".")[0]
    #print(filename)
    sampleend = f"/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/converted_data/three-sixty/{filename}.json"
    #if os.path.exists(sampleend):
    #    continue
    #else:
    process_file_with_timeout(file, timeout_seconds=20 * 60 * 60)  # Set timeout to 10 seconds (adjust as needed)
    import json
    with open("errors.json", "w") as outfile: 
        json.dump(errors, outfile)