#Script which loads data from github to json
import requests
import pandas as pd
# URL of the raw file on GitHub
matches = pd.read_json("/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/matches/53/106.json")["match_id"]
three_sixty = "https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/three-sixty/"
event = "https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/events/"
lineups = "https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/lineups/"

local_three_sixty = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/three-sixty/"
local_event = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/events/"
local_lineups = "/home/lz80/rdf/sp161/shared/soccer-decision-making/womens_euro/lineups/"
# Make a GET request to fetch the raw file content

all_files = [three_sixty, event, lineups]
all_dest = [local_three_sixty, local_event, local_lineups]
for match in matches:
    print(f"Downloading {match}")
    for file in [0,1,2]:

        file_url = all_files[file] + str(match) + ".json"
        dest_path = all_dest[file] + str(match) + ".json"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(dest_path, 'wb') as file:
                file.write(response.content)
            #print("File downloaded successfully!")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")