# %%
#Helper script which converts buli game data to statsbomb game data - legacy code
import pandas as pd
import xml.etree.ElementTree as ET
import pandas as pd
import json
needed = {"match_id": "NullError", "match_date": "NullError", 
          "kick_off": "No Change", "competition": "NullError", "season": "NullError",
            "home_team": "NullError", "away_team": "NullError",
           "home_score": "No Change", 
          "away_score": "No Change", "match_status": "No Change", "match_status_360": 
          "No Change", "last_updated": "No Change", "last_updated_360": "No Change",
          "metadata": "No Change", "match_week": "No Change", "competition_stage": "No Change",
            "stadium": "No Change", "referee": "No Change"}
def convert_gamedata_lineups(gamedata_path, players_path, lineups_outpath, gamedata_outpath):
    given_gamedata = pd.read_csv(gamedata_path, encoding='latin-1', on_bad_lines='skip')#
    statsbomb_gamedata = pd.read_json("/home/lz80/un-xPass/stores/datasets/sample_files/106.json")

    # %%
    nulls = []
    nochange = []
    for col in needed:
        if needed[col] == "NullError":
            nulls.append(col)
        else:
            nochange.append(col)
    nulls

    # %%
    """convert matches"""
    given_gamedata["match_id"] = given_gamedata["MatchId"]
    given_gamedata["match_date"] = pd.to_datetime(given_gamedata["StartDate"]).dt.date
    given_gamedata["competition"] = [{'competition_id': 9,
    'country_name': 'Germany',
    'competition_name': '1. Bundesliga'}] * given_gamedata.shape[0]
    given_gamedata["season"] = [{'season_id': 281, 'season_name': '2023/2024'}] * given_gamedata.shape[0]
    given_gamedata["home_team"] = given_gamedata.apply(lambda d: {"home_team_id":d["HomeTeamId"]}, axis = 1)
    given_gamedata["away_team"] = given_gamedata.apply(lambda d: {"away_team_id":d["GuestTeamId"]}, axis = 1)
    given_gamedata[nochange] = None
    gamedata_converted = given_gamedata[statsbomb_gamedata.columns]
    gamedata_converted["match_date"] = gamedata_converted["match_date"].astype(str)


    # %%

    def load_players(path, to_dict):
        tree = ET.parse(path)  # Replace with your XML file path
        root = tree.getroot()

        data = []
        for root_elem in root.findall('.//Team'):
            TeamName = root_elem.attrib.get('TeamName')
            Formation = root_elem.attrib.get('LineUp')
            TeamId = root_elem.attrib.get('TeamId')
            Role = root_elem.attrib.get('Role')
            for elem in root_elem.findall('.//Player'):  
                entry = {}
                entry = {
                        "Role": Role,
                        "TeamName": TeamName,
                        "TeamId": TeamId,
                        "Formation":Formation,
                        #"PersonId": PersonId,
                        "Starting": elem.attrib.get('Starting'),
                        'PersonId': elem.attrib.get('PersonId'),
                        'FirstName': elem.attrib.get('FirstName'),
                        'LastName': elem.attrib.get('LastName'),
                        'PlayingPosition': elem.attrib.get('PlayingPosition'),
                        'ShirtNumber':elem.attrib.get('ShirtNumber')
                }
                data.append(entry)

        df = pd.DataFrame(data)
        if to_dict:
            df = pd.Series(df["PlayingPosition"].values,index=df["PersonId"]).to_dict()
        return df


    # %%
    """
    Competition: {'competition_id': 9,
    'country_name': 'Germany',
    'competition_name': '1. Bundesliga'}
    season: {'season_id': 281, 'season_name': '2023/2024'}
    """

    # %%
    #lineups: ['player_id', 'player_name', 'player_nickname', 'jersey_number']
    #lineups_sb = pd.read_json(lineups_path)
    players_buli = load_players(players_path, False)
    #player_id - num/str, player_name - str, player_nickname: None, jersey_number - int


    # %%
    """
    Lineup Conversion
    """
    players_buli["player_dict"] = players_buli.apply(lambda d: {"player_id":d["PersonId"], "player_name":d["FirstName"] + " " + d["LastName"], "player_nickname":None, "jersey_number": int(d["ShirtNumber"])}, axis = 1)
    team1 = players_buli["TeamId"].unique()[0]
    team2 = players_buli["TeamId"].unique()[1]
    team1_name = players_buli[players_buli["TeamId"] == team1]["TeamName"].reset_index(drop = True).loc[0]
    team2_name = players_buli[players_buli["TeamId"] == team2]["TeamName"].reset_index(drop = True).loc[0]
    team1_lineup = list(players_buli[players_buli["TeamId"]== team1]["player_dict"])
    team2_lineup = list(players_buli[players_buli["TeamId"]== team2]["player_dict"])
    lineups_converted = pd.DataFrame()
    lineups_converted["team_id"] = [team1, team2]
    lineups_converted["team_name"] = [team1_name, team2_name]
    lineups_converted["lineup"] = [team1_lineup, team2_lineup]


    # %%
    def convert_to_json(df, path):
            json_str = df.to_json(orient='records')
            json_data = json.loads(json_str)
            cleaned_data = json_data
            with open(path, 'w') as json_file:
                    json.dump(cleaned_data, json_file, indent=2)
    convert_to_json(lineups_converted, lineups_outpath)
    convert_to_json(gamedata_converted, gamedata_outpath)


