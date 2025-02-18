# %%
#Converts buli tracking and event to statsbomb data(v2) - contains flips
import xml.etree.ElementTree as ET
import pandas as pd
def convert_event_and_tracking(filename, players_p, tracking, together_csv, event_p, matchplan, event_outpath, tracking_outpath):
    def load_tracking(path):
        tree = ET.parse(path)  # Replace with your XML file path
        root = tree.getroot()

        data = []
        for root_elem in root.findall('.//FrameSet'):
            GameSection =  root_elem.attrib.get('GameSection')
            MatchId = root_elem.attrib.get('MatchId')
            TeamId = root_elem.attrib.get('TeamId')
            PersonId = root_elem.attrib.get('PersonId')
            for elem in root_elem.findall('.//Frame'):  
                entry = {}
                entry = {
                        "GameSection": GameSection,
                        "MatchId": MatchId,
                        "TeamId": TeamId,
                        "PersonId": PersonId,
                        'N': elem.attrib.get('N'),
                        'T': elem.attrib.get('T'),
                        'X': elem.attrib.get('X'),
                        'Y': elem.attrib.get('Y'),
                        'D': elem.attrib.get('D'),
                        'S': elem.attrib.get('S'),
                        'A': elem.attrib.get('A'),
                        'M': elem.attrib.get('M')
                }
                data.append(entry)

        df = pd.DataFrame(data)
        return df


    # %%
    import xml.etree.ElementTree as ET
    import pandas as pd
    def load_event(path):
        # Load and parse the XML file
        tree = ET.parse(path)  # Replace 'events.xml' with your file path
        root = tree.getroot()

        # Initialize an empty list to store the parsed data
        data = []

        # Function to recursively parse the XML tree
        def parse_event(event_element, parent_data):
            for child in event_element:
                # Combine parent data with child attributes
                child_data = {**parent_data, **child.attrib}

                # If the child has further sub-elements, recurse
                if list(child):
                    parse_event(child, child_data)
                else:
                    # If the child is a leaf node, append the data to the list
                    data.append(child_data)

        # Loop through each Event in the XML
        for event in root.findall('Event'):
            event_data = event.attrib  # Get attributes of the Event node
            for child in event:
                #print(child)
                if child.tag == "Play":
                    for subchild in child:
                        event_data["EventType"] = subchild.tag
                else:
                    event_data["EventType"] = child.tag
            parse_event(event, event_data)

        # Convert the list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame
        return df

    # %%
    import xml.etree.ElementTree as ET
    import pandas as pd

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
    players = load_players(players_p, True)
    players#TW = goalkeeper

    # %%
    #7, 8, 13, 14, 15, 25
    position_map = {"TW": {"id": 1, "name": "GoalKeeper"}, 
                    "OLM": {"id": 20, "name": "Left Attacking Midfielder"}, 
                    "ZO": {"id": 19, 'name': "Center Attacking Midfielder"},
                    "STZ": {"id": 23, 'name':"Striker"},
                    "DMR": {"id": 9, 'name':"Right Defensive Midfielder"},
                    "DML": {"id": 11, 'name':"Left Defensive Midfielder"},
                    "DMZ": {"id": 10, 'name':"Center Defensive Midfielder"},
                    "IVL": {"id":5, 'name':"Left Center Back"},
                    "IVR": {"id": 3, 'name':"Right Center Back"},
                    "IVZ": {"id":4, "name":"Center Back"},
                    "LM": {"id": 16, "name":"Left Midfielder"},
                    "LV": {"id": 6, "name":"Left Back"},
                    "OHL": {"id": 21, "name":"Left Winger"},
                    "OHR": {"id": 17, 'name':"Right Winger"},
                    "ORM": {"id":18, 'name':'Right Offensive Midfielder'},
                    "RM": {"id":12, 'name':"Right Midfielder"},
                    "RV": {"id":2, "name":"Right Back"},
                    "STL": {"id": 24, "name":"Left Center Forward"},
                    "STR": {"id": 22, "name":"Right Center Forward"},
                    "RAV": {"id":7, "name":"Right Wing Back"},
                    "LAV": {"id":8, "name":"Left Wing Back"},
                    "RZM": {"id":13, "name":"Right Center Midfielder"},
                    "ZM": {"id": 14, "name": "Center Midfielder"},
                    "LZM": {"id":15, "name":"Left Center Midfielder"},
                    "SS": {"id": 25, "name":"Shadow Striker"}



                    
                    }

    # %%
    tracking = load_tracking(tracking)

    # %%
    def resolve_duplicates(group):
        # Filter for Pass SUBTYPE
        pass_rows = group[group['SUBTYPE'] == 'Pass']
        
        if not pass_rows.empty:
            # Filter for Pass rows where XRec is not None
            pass_with_recipient = pass_rows[pass_rows['XRec'].notna()]
            
            if not pass_with_recipient.empty:
                # If we have pass rows with non-None XRec, take the first
                return pass_with_recipient.iloc[0]
            else:
                # Otherwise, take the first row within Pass group (even if XRec is None)
                return pass_rows.iloc[0]
        
        # If there are no Pass rows, just take the first row of the group
        return group.iloc[0]

    # %%
    together = pd.read_csv(together_csv,sep = ';', encoding='latin-1', on_bad_lines='skip')
    together = together.groupby('FRAME_NUMBER', group_keys=False).apply(resolve_duplicates).reset_index(drop = True)
    together = together.sort_values(by = "FRAME_NUMBER")
    merged = together[["EVENT_ID", "MUID", "PUID1", "CUID1", "PUID2", "CUID2", "FRAME_NUMBER"]]

    # %%
    event_flip = load_event(event_p)
    event_flip["TeamRight"]

    # %%
    import numpy as np
    event_flip = load_event(event_p)
    event_flip = event_flip[event_flip["EventType"] != "Delete"].reset_index(drop = True)
    event_flip["TeamRight"] = event_flip["TeamRight"].fillna(method = 'ffill')
    event_flip["TeamLeft"] = event_flip["TeamLeft"].fillna(method = 'ffill')
    event_flip["NeedsFlip"] = np.where(event_flip["TeamRight"] != event_flip["Team"], True, False)
    event_flip = event_flip[["EventId", "NeedsFlip"]]
    event_flip["EventId"] = event_flip["EventId"].astype(int)
    together = pd.merge(together, event_flip, left_on = "EVENT_ID", right_on = "EventId")

    # %%
    import numpy as np
    tracking["N"] = tracking["N"].astype(int)
    tracking_together = pd.merge(tracking, together, left_on = "N", right_on = "FRAME_NUMBER")[["EVENT_ID", "TeamId", "PersonId", "N", "X", "Y", "CUID1", "PUID1", 'NeedsFlip']]
    tracking_together = tracking_together[tracking_together["TeamId"] != "BALL"]
    tracking_together["Player_Position"] = tracking_together["PersonId"].map(players)
    tracking_together["goalkeeper"] = np.where(tracking_together["Player_Position"] == "TW", True, False)
    tracking_together["teammate"] = np.where(tracking_together["TeamId"] == tracking_together["CUID1"], True, False)
    tracking_together["actor"] = np.where(tracking_together["PersonId"] == tracking_together["PUID1"], True, False)
    tracking_together["X_translated"] = 120 - (1.09361 * tracking_together["X"].astype(float) + 60)
    tracking_together["Y_translated"] = 80 - (-1.09361 * tracking_together["Y"].astype(float) + 40)#???
    tracking_together["X_translated"] = np.where(tracking_together["NeedsFlip"], 120 - tracking_together["X_translated"], tracking_together["X_translated"])
    # Function to convert the DataFrame
    def convert_to_frame_based(df):
        # Group by 'frame'
        event_group = df.groupby('EVENT_ID')
        
        # Initialize an empty list to store the new rows
        event_data = []

        # Iterate through each frame group
        for event, group in event_group:
            # Create the list of dictionaries for each frame
            player_dicts = group.apply(lambda row: {
                'location': [row['X_translated'],row['Y_translated']],
                'teammate': row['teammate'],
                'keeper': row['goalkeeper'],
                'actor': row['actor']
            }, axis=1).tolist()
            
            # Append the frame data as a dictionary
            event_data.append({'event_uuid': event, 'freeze_frame': player_dicts})
        
        # Create a new DataFrame with each frame and its list of player dictionaries
        return pd.DataFrame(event_data)

    # Convert the DataFrame
    tracking_converted = convert_to_frame_based(tracking_together)
    tracking_converted["visible_area"] = [[0,0, 120, 0, 120, 80, 0, 80, 0,0]] * tracking_converted.shape[0]
    tracking_converted = tracking_converted[["event_uuid", "visible_area", "freeze_frame"]]
    tracking_converted

    # %%
    #threesixty = pd.read_json("/home/lz80/un-xPass/stores/bulidata/3795506-360.json")
    #threesixty#360 freeze-frame: teammate(boolean), actor(boolean), keeper(boolean), location([x,y])
    #event_id = string, visible_area is everywhere i guess - convert x, y probably
    #threesixty["freeze_frame"].loc[0]

    # %%
    statsbomb_event = pd.read_json("/home/lz80/un-xPass/stores/datasets/sample_files/sample_statsbomb.json")
    #statsbomb_event#seem similar?

    # %%


    # %%
    matchplan = pd.read_xml(matchplan, xpath = ".//Fixture")
    matchplan

    # %%
    event = load_event(event_p)
    event = event[event["EventType"] != "Delete"].reset_index(drop = True)
    #event

    # %%
    event["TeamRight"] = event["TeamRight"].fillna(method = 'ffill')
    event["NeedsFlip"] = np.where(event["TeamRight"] != event["Team"], True, False)
    #attacking team goes to the right
    #if teamRight and attacking - flip
    #if teamLeft and attacking - do not flip

    #event["EventId"]

    # %%
    #event[(event["NeedsFlip_x"]) & (event["location"].notna())]#[["location", "EventType"]]
    #event[["location", "NeedsFlip_x"]].loc[0:5]

    # %%
    import numpy as np
    """
    Calculating Time
    """
    event["EventTime"] = pd.to_datetime(event["EventTime"])
    #event = event.sort_values(by = ["EventTime"]).reset_index(drop = True)
    secondhalf = event[(event["GameSection"] == "secondHalf") & (event["EventType"] == "KickOff")].index[0]
    #print(secondhalf)
    print(event)
    starttime = event.loc[0]["EventTime"]
    halftime = event.loc[secondhalf]["EventTime"]
    event["period"] = np.where(event.index >= secondhalf, 2, 1)
    event["timestamp"] = np.where(event["period"] == 1, event["EventTime"] - starttime, event["EventTime"] - halftime)#take another look at this
    event["date"] = event["EventTime"].dt.date
    event["second"] = event["timestamp"].astype('timedelta64[s]')
    event["minute"] = event["second"] // 60
    event["minute"] = np.where(event["period"] == 2, event["minute"] + 46, event["minute"])
    event["second"] = event["second"] % 60
    event["duration"] = event["EventTime"] - event["EventTime"].shift(1)
    event["timestamp"] = pd.to_datetime(event["date"]) + event["timestamp"]
    #event[["time","minute","second", "period"]]
    """
    Index
    """
    event["index"] = event.index


    # %%
    """
    Calculating Possession
    """
    matchid = filename
    #event["Evaluation"]#possession changes -> unsuccessful passes, tackles, shots - there is probably a better way to do this - though it might not matter
    #event[event["EventType"] == "ShotAtGoal"]["Team"]
    event["Rebounded"] = np.where((event["EventType"] == "ShotAtGoal") & (event["Team"] == event["Team"].shift(-1)), 1, 0)
    event["Changed_Possession"] = np.where((event["Rebounded"] == 1) | (event["Evaluation"]=="unsuccessful") | (event["PossessionChange"] == "true"),1,0)
    event["possession"] = event["Changed_Possession"].cumsum() + 1
    event["possession"] = np.where(event["EventType"] == "KickOff", event["possession"] + 1, event["possession"])
    event["possession_team"] = event["Team"]
    event["possession_team"] = np.where(event["EventType"] == "TacklingGame", event["WinnerTeam"], event["possession_team"])

    event["possession_team"] = np.where(event["EventType"] == "Foul", event["TeamFouled"], event["possession_team"])
    event["possession_team"] = np.where(event["EventType"] == "FinalWhistle", event["possession_team"].shift(1), event["possession_team"])
    while event[event["possession_team"].isna()].shape[0] != 0:
        event["possession_team"] = np.where(event["possession_team"].isna(), event["possession_team"].shift(1), event["possession_team"])
    gameplan = matchplan.loc[np.where(matchplan["MatchId"] == matchid)[0][0]]
    event["TeamName"] = np.where(event["possession_team"] == gameplan["HomeTeamId"], gameplan["HomeTeamName"], gameplan["GuestTeamName"])
    event["possession_team"] = event.apply(lambda row: {'id': row['possession_team'], 'name': row['TeamName'] }, axis=1)
    event["team"] = event['possession_team']

    # %%
    """ 
    play pattern
    """
    event["GK_Event"] = np.where(event["GoalKeeperAction"].isin(["throwOut", "punt"]), "GoalKeeperAction", event["EventType"])
    event["starting_event"] = event.groupby('possession')['GK_Event'].transform('first')
    #event["starting_event"] = np.where(event["GoalKeeperAction"].isin(["throwOut", "punt"]), "GoalKeeperAction", event["starting_event"])
    eventmap = {"CornerKick": {'id': 2, 'name': 'From Corner'}, 'FreeKick': {'id': 3, 'name': 'From Free Kick'}, 
                'ThrowIn': {'id': 4, 'name': 'From Throw In'}, 'GoalKick': {'id': 7, 'name': 'From Goal Kick'},
                'GoalKeeperAction': {'id': 8, 'name': 'From Keeper'}, 'KickOff':  {'id': 9, 'name': 'From Kick Off'}}#can include counterattack maybe? see counterattack column
    event["play_pattern"]= event["starting_event"].map(eventmap).fillna(value = "{'id': 1, 'name': 'Regular Play'}")
    event["play_pattern"]

    #event[event["starting_event"] == "ShotAtGoal"]
    #event.loc[930:937]

    # %%
    event["EventUpdated"] = np.where(event["EventType"].isin(["FreeKick", "GoalKick", "CornerKick"]), event["EventType"].shift(-1), event["EventType"])
    while event[event["EventUpdated"].isin(["FreeKick", "GoalKick", "CornerKick"])].shape[0] > 0:
        event["EventUpdated"] = np.where(event["EventUpdated"].isin(["FreeKick", "GoalKick", "CornerKick"]), event["EventUpdated"].shift(-1), event["EventUpdated"])
    new_events = {
        "ShotAtGoal": {'id': 16, 'name': 'Shot'},
        "Foul": {'id': 21, 'name': 'Foul Won'},
        "TacklingGame": {'id': 4, 'name': 'Duel'},
        "ThrowIn": {'id': 30, 'name': 'Pass'},
        "OtherBallAction": {'id': 1, 'name': 'OtherBallAction'},
        "Delete": {'id': -1, 'name': 'Delete'},
        "Offside": {'id': 8, 'name': 'Offside'},
        "Cross": {'id': 30, 'name': 'Pass'},
        "Pass": {'id': 30, 'name': 'Pass'},
        "Caution": {'id': 24, 'name': 'Bad Behaviour'},
        "FairPlay": {'id': 41, 'name': 'Referee Ball-Drop'},#maybe a drop ball? #41 / “Referee Ball-Drop”
        "ChanceWithoutShot": {'id': 30, 'name': 'Pass'},
        "FinalWhistle": {'id': 34, 'name': 'Half End'},
        "Substitution": {'id': 19, 'name': 'Substitution'},
        "Run": {'id': 43, 'name': 'Carry'},
        "Nutmeg": {'id': 43, 'name': 'Carry'},
        "BallDeflection": {'id': 10, 'name': 'Interception'},
        "BallClaiming": {'id': 2, 'name': 'Ball Recovery'},
        "KickOff" : {'id': 30, 'name': 'Pass'},
        "GoalDisallowed" : {'id': 1, 'name': 'OtherBallAction'}
    }
    event["type"] = event["EventUpdated"].map(new_events)
    event["tactics"] = None#needs to be added with starting xi!!!!
    event["related_events"] = np.where(event["EventId"].notna(), event['EventId'].shift(-1).apply(lambda x: [x]), None)
    event["buli_Player"] = event["Player"]
    event["player_id"] = np.where(event["buli_Player"].notna(), event["buli_Player"], np.where(event["EventType"] == "TacklingGame", event["Winner"], event["PlayerIn"]))
    playernames = load_players(players_p, False)
    playernames["name"] = playernames["FirstName"] + " " + playernames["LastName"]
    playername_dict = pd.Series(playernames["name"].values,index=playernames["PersonId"]).to_dict()
    event["player_name"] = event["player_id"].map(playername_dict)
    event["player"] = event.apply(lambda d: {"id": d["player_id"], "name":d["player_name"]}, axis = 1)
    event["position"] = event["Player"].map(players).map(position_map)
    event["EventId"] = event["EventId"].astype(int)
    together["EVENT_ID"] = together["EVENT_ID"].astype(int)
    event = pd.merge(event, together[["EVENT_ID", "FRAME_NUMBER"]], left_on = "EventId", right_on = "EVENT_ID", how = "left")

    #event["X_EVENT"] = event["X_EVENT"].str.replace(",", ".")
    #event["Y_EVENT"] = event["Y_EVENT"].str.replace(",", ".")
    #event["X-Position_translated"] = 1.09361 * event["X_EVENT"].astype(float) + 60
    #event["Y-Position_translated"] = 1.09361 * -1 * event["Y_EVENT"].astype(float) + 40
    #event["location"] = list(event[["X-Position_translated", "Y-Position_translated"]].astype(float).values)#use tracking data probably?
    #event['location'] = event['location'].apply(lambda x: None if any(pd.isna(i) for i in x) else x)#check these again

    # %%
    ball_locs = tracking[tracking["TeamId"] == "BALL"][["N", "X", "Y"]].rename(columns = {"X":"ball_x", "Y": "ball_y", "N":"Frame_Number"})
    event = pd.merge(event, ball_locs, left_on = "FRAME_NUMBER", right_on = "Frame_Number", how = "left")
    event["ball_x_translated"] = 120 - (1.09361 * event["ball_x"].astype(float) + 60)
    event["ball_y_translated"] =80 - (-1.09361 * event["ball_y"].astype(float) + 40)
    event["ball_x_translated"] = np.where(event["NeedsFlip"] & event["ball_x_translated"].notna(), 120 - event["ball_x_translated"], event["ball_x_translated"])
    event["location"] = list(event[["ball_x_translated", "ball_y_translated"]].astype(float).values)
    event['location'] = event['location'].apply(lambda x: None if any(pd.isna(i) for i in x) else x)

    # %%
    event[event["Frame_Number"] == 68452]["location"]

    # %%
    """Pass Column"""
    together["YRec"] = together["YRec"].str.replace(",", ".")
    together["XRec"] = together["XRec"].str.replace(",", ".")
    together["Dist_of_pass"] = together["Dist_of_pass"].str.replace(",", ".")
    together["Angle_of_pass"] = together["Angle_of_pass"].str.replace(",", ".")
    event["EventId"] = event["EventId"].astype("float")
    #together[["XRec", "YRec"]].loc[0:40]
    #together[together["XRec"] == '33,43']

    # %%
    together = together.sort_values(by = "FRAME_NUMBER").reset_index(drop = True)
    def find_next_different(row_idx, col):
        current_value = together.loc[row_idx, col]
        # Loop through the subsequent values to find the next different one
        for next_idx in range(row_idx + 1, len(together)):
            if together.loc[next_idx, col] != current_value:
                return together.loc[next_idx, col]
        return None 
    together = together.sort_values(by = "FRAME_NUMBER")
    together['NEXT_FRAME_NUMBER'] = together.index.to_series().apply(find_next_different, col='FRAME_NUMBER')
    needs = together[(together["SUBTYPE"] == "Pass") & (together["XRec"].isna()) & (together["PUID2"].notna())].reset_index(drop = True)
    unsuccessful = []
    for i in needs.index:#finding closest is not perfect - receiver can carry - first time when below a certain threshold??
        print(f"{i + 1} / {len(needs.index)}")
        row = needs.loc[i]
        init = int(row["FRAME_NUMBER"])
        if(row["NEXT_FRAME_NUMBER"] != row["NEXT_FRAME_NUMBER"]):
            continue #last pass - no rec?
        last = int(row["NEXT_FRAME_NUMBER"])
        print(init, last)
        eventid = row["EVENT_ID"]#
        print(eventid)
        recipient = row["PUID2"]
        recipient_coords = tracking[(tracking["N"].astype(int) >= init) & (tracking["N"].astype(int) <= last) & (tracking["PersonId"] == recipient)].reset_index(drop = True)
        #print(recipient_coords.shape)
        recipient_coords = recipient_coords.drop_duplicates(subset=['N'])
        ball_coords = tracking[(tracking["N"].astype(int) >= init) & (tracking["N"].astype(int) <= last) & (tracking["PersonId"] == "DFL-OBJ-0000XT")].reset_index(drop = True)
        ball_coords = ball_coords.drop_duplicates(subset=['N'])
        dist = ((recipient_coords["X"].astype(float)- ball_coords["X"].astype(float))**2 + (recipient_coords["Y"].astype(float) - ball_coords["Y"].astype(float))**2)**(1/2)
        #print(recipient_coords["X"].astype(float)- ball_coords["X"].astype(float))
        #passes at end of half cause errors
        print(len(ball_coords["N"].unique()))
        rel_df = pd.DataFrame(data = {"FRAME_NUMBER" : ball_coords["N"], "ball_x": ball_coords["X"], "ball_y":ball_coords["Y"], "dist":dist})
        rel_df["EVENT_ID"] = eventid
        min_idx = rel_df[["dist"]].idxmin()
        rel_df_new = rel_df.loc[min_idx]
        print(eventid)
       #print(rel_df_new)
        #18477700001638
        unsuccessful.append(rel_df_new)
    all_unsuccess = pd.concat(unsuccessful)


    # %%
    together = pd.merge(together, all_unsuccess, left_on = "EVENT_ID", right_on = "EVENT_ID", how = "left")
    together["XRec"] = np.where(together["ball_x"].notna(), together["ball_x"], together["XRec"])
    together["YRec"] = np.where(together["ball_y"].notna(),  together["ball_y"], together["YRec"])
    #together["XRec"] = np.where()
    #together["XRec"] = np.where((together["SUBTYPE"] == "Pass") & (together["XRec"].isna()) & (together["PUID2"].isna()), together["ball_x"].shift(-1), together["XRec"])
    #together["YRec"] = np.where((together["SUBTYPE"] == "Pass") & (together["YRec"].isna()) & (together["PUID2"].isna()), together["ball_y"].shift(-1), together["YRec"])
    together["XRec"] = together["XRec"].str.replace(",", ".")
    together["YRec"] = together["YRec"].str.replace(",", ".")#18453000000640

    # %%
    together["XRec"] = 120 - (1.09361 * together["XRec"].astype(float) + 60)
    together["YRec"] = 80 - (1.09361 * -1 * together["YRec"].astype(float) + 40)#need to flip again?

    # %%
    event = pd.merge(event, together, left_on = "EventId", right_on = "EVENT_ID", how = "left")
    
    # %%
    event = event.sort_values(by = "FRAME_NUMBER")
    #event = pd.merge(event, tracking, left_on = ["FRAME_NUMBER", "Recipient"], right_on = ["N", "PersonId"], how="left").drop(columns = ["N", "PersonId"])
    import math
    event["Angle_of_pass_new"] = event["Angle_of_pass"].str.replace(",", ".")
    event["angle"] = event["Angle_of_pass_new"].astype(float) * math.pi / 180
    event["length"] = np.sqrt((event["ball_x_translated"].astype(float) - event["XRec"].astype(float))**2 + (event["ball_y_translated"].astype(float) - event["YRec"].astype(float))**2)

    event['name'] = event["Recipient"].map(playername_dict)

    pass_map = {"Corner":{'id': 61, 'name': 'Corner'}, "FreeKick": {'id': 62, 'name': 'Free Off'},
                "GoalKick":{'id': 63, 'name': 'Goal Kick'}, "KickOff": {'id': 65, 'name': 'Kick Off'}, "ThrowIn":{'id': 67, 'name': 'Throw-in'}}

    event['pass_recipient_dict'] = event.apply(
        lambda row: {'id': row['Recipient'], 'name': row['name']} 
                    if not (pd.isna(row['Recipient']) and pd.isna(row['name'])) 
                    else np.nan, axis=1
    )
    event["XRec"] = np.where(event["NeedsFlip_x"] & event["XRec"].notna(), 120 - event["XRec"], event["XRec"])
    event["end_location"] = list(event[["XRec","YRec"]].astype(float).values)
    event['end_location'] = event['end_location'].apply(lambda x: None if any(pd.isna(i) for i in x) else x)
    event['pass_type'] = event["EventType"].map(pass_map)
    event["outcome"] = np.where(event["Evaluation"] == "unsuccessful", {"id":9,"name":"Incomplete"}, None)
    
    # Perform the shift and bfill operation
    bfilled_values = event["location"].shift(-1).bfill()

    # Adjust the first element of the list conditionally
    adjusted_bfilled_values = bfilled_values.copy()
    print(bfilled_values.to_csv("wtf.csv"))
    for idx, value in bfilled_values.iteritems():
        if isinstance(value, np.ndarray) and len(value) > 0:
            # Check if the current "NeedsFlip" differs from the "NeedsFlip" of the row with the bfilled value
            next_idx = idx + 1
            iter = 0
            breakall = False
            while next_idx not in event.index:
                if(iter > 20):
                    breakall = True
                    break
                next_idx = next_idx + 1
                iter = iter + 1
            if breakall:
                continue
            if next_idx <= event.shape[0] and event.loc[next_idx]["NeedsFlip_x"] != event.loc[idx]["NeedsFlip_x"]:
                adjusted_bfilled_values[idx] = [120 - value[0],value[1]]
            else:
                adjusted_bfilled_values[idx] = value
    test = event[(event["end_location"].isna()) & (event["SUBTYPE"] == "Pass")]
    test.to_csv("wtf_2.csv")
    # Update 'end_location' using the adjusted bfilled values
    event["end_location"] = np.where(
        (event["end_location"].isna()) & (event["SUBTYPE"] == "Pass"),
        adjusted_bfilled_values,
        event["end_location"]  # Retain the existing value if the condition is not met
    )

    
    #event["end_location"] = np.where(
    #(event["end_location"].isna()) & (event["SUBTYPE"] == "Pass"),
    #event["location"].shift(-1).bfill(),
    #event["end_location"]
    #)
    event["pass"] = event.apply(lambda row: {"recipient":row["pass_recipient_dict"], "length":row["length"], "angle":row["angle"],
                            'height': {'id': 1, 'name': 'Ground Pass'}, "end_location": row["end_location"], "type": row['pass_type'], "body_part":{'id': 40, 'name': 'Right Foot'}, 'outcome': row["outcome"]},axis=1)
    event["pass"] = event["pass"].apply(lambda d: {k: v for k, v in d.items() if v is not np.nan and v is not None} if isinstance(d, dict) else d)
    #need to translate coordinates..., also recipients - check this method!! I wrote it when I was tired


    # %%
    #Carry - need endlocation - uses start location of next event - not sure best way but I think this is the only way
    event["next_x"] = event["ball_x_translated"].shift(-1)
    event["next_y"] = event["ball_y_translated"].shift(-1)
    event["carry"] = np.where(event["type"] == {"id": 43, "name":"Carry"}, event.apply(lambda row: {"end_location":[row["next_x"],row["next_y"]]}, axis=1), None)

    # %%
    #statsbomb_event[statsbomb_event['under_pressure'].notna()]['under_pressure']
    event['under_pressure'] = None#essentially dummy data to make sure the json is happy
    event["ball_receipt"] = None
    #'counterpress', 'interception', 'clearance'
    event['counterpress'] = None
    event['interception'] = None
    event['clearance'] = None
    event["counterpress"] = None#
    #event[""]#fill with Nones


    # %%
    """Duel"""
    #statsbomb_event[statsbomb_event["duel"].notna()]["duel"].loc[122]
    event["duel"] = np.where(event["PossessionChange"].astype(str) == "true", {'type': {'id': 11, 'name': 'Tackle'},
    'outcome': {'id': 4, 'name': 'Won'}}, None)
    event["duel"] = np.where(event["PossessionChange"].astype(str) == "false", {'type': {'id': 11, 'name': 'Tackle'}, #this is a silly way to do this, but it works!! - not really comprehensive but I don't think this will affect
    'outcome': {'id': 1, 'name': 'Lost'}}, None)
    event = event.loc[:,~event.columns.duplicated()].copy()

    # %%
    """
    Shot
    """

    #event["SUBTYPE"].unique()
    #SuccessfulShot; BlocketShot, ShotWide, ShotWoodWork, SavedShot
    outcomes = {"BlockedShot": {"id": 96, "name":"Blocked"}, "ShotWide":{"id":98, "name":"Off T"},
                "ShotWoodWork":{"id":99, "name":"Post"}, "SavedShot":{"id":100, "name":"Saved"}, 
                "SuccessfulShot":{"id":97, "name":"Goal"}, "OtherShot":{"id":98, "name":"Off T"}}
    end_location_map = {1:[120,2,37], 2: [120,2,38], 3:[120,2,40], 4:[120,2,41], 5:[120,2,43],
                        6:[120,1,37], 7:[120,1,38], 8:[120,1,40], 9:[120,1,41], 10:[120,1,43],
                        11:[120,0,37], 12: [120,0,38], 13:[120,0, 40], 14:[120, 0, 41], 15:[120, 0, 43]}

    #'technique': {'id': 93, 'name': 'Normal'}
    #event["TypeOfShot"].unique()
    if("GoalZone" in event.columns):   
        event["shot_end_location"] = event["GoalZone"].astype(float).map(end_location_map)
    else:
        event["shot_end_location"] = None
    def update_shot_end_location(row):
        # Check if end_x <= 60 and shot_end_location is a list (not NaN)
        if row['ball_x_translated'] <= 60 and isinstance(row['shot_end_location'], list):
            row['shot_end_location'][0] = 0  # Modify the first element to 0
        return row
    event = event.apply(update_shot_end_location, axis=1)
    def replace_shot_end_location(row):
        if row['SUBTYPE'] in ["BlocketShot", "ShotWide", "ShotWoodWork", "SavedShot", "OtherShot"]:
            return [row['next_x'], row['next_y']]
        return row['shot_end_location']

    event['shot_end_location'] = event.apply(replace_shot_end_location, axis=1)
    event['shot_outcomes'] = event["SUBTYPE"].map(outcomes)
    event["xG_x"] = event["xG_x"].str.replace(",", ".")
    event["xG_x"] = event["xG_x"].astype(float)
    def consolidate_row(row):
        
            # Create a dictionary with non-null values
        return {
                'end_location': row['shot_end_location'],
                'statsbomb_xg': row['xG_x'],
                'outcome': row['shot_outcomes']
            }
        
    event['shot'] = event.apply(consolidate_row, axis=1)
    event["shot"] = event["shot"].apply(lambda d: None if np.nan in d.values() else d)

    # %%
    """
    Goalkeeper - just need "type"
    """
    outcomes = {"BlockedShot": {"id": 33, "name":"Shot Faced"}, "ShotWide":{"id": 33, "name":"Shot Faced"},
                "ShotWoodWork":{"id": 33, "name":"Shot Faced"}, "SavedShot":{"id":31, "name":"Save"}, 
                "SuccessfulShot":{"id":26, "name":"Goal Conceded"}}
    event["goalkeeper"] = event["SUBTYPE"].map(outcomes)
    def make_gk(row):
        return {'type': row["goalkeeper"]}
    event["goalkeeper"] = event.apply(make_gk, axis = 1)
    event["goalkeeper"] = event["goalkeeper"].apply(lambda d: None if np.nan in d.values() else d)
    event["type"] = np.where(event["goalkeeper"].notna(), {
        "id": 23,
        "name": "Goal Keeper"
        }, event["type"])

    # %%
    event[['off_camera', 'dribble', 'foul_committed', 'foul_won', 'out',
        'ball_recovery', 'miscontrol', 'substitution', 'injury_stoppage']] = None
    event["id"] = event["EventId_x"]
    converted_event = event[statsbomb_event.columns]
    converted_event = converted_event[converted_event["id"].isin(together["EVENT_ID"])]
    # %%

    perfect = ['id', 'index' ,'type',
    'period',
    'timestamp',
    'team',
    'location',
    'pass',
    'carry',
    'duel',
    'shot',
    'goalkeeper']
    imperfect = ['minute',#if replaced with copies
    'second',
    'possession',
    'possession_team',
    'play_pattern',
    'duration',
    'tactics',
    'related_events',
    'player',
    'position',
    'under_pressure',
    'ball_receipt',
    'counterpress',
    'interception',
    'clearance',
    'off_camera',
    'dribble',
    'foul_committed',
    'foul_won',
    'out',
    'ball_recovery',
    'miscontrol',
    'substitution',
    'injury_stoppage']


    # %%
    nulls = {"id": "NullError", "index": "No Change", "period": 66, "timestamp": 66, "minute": "No Change", "second": "No Change", "type": "NullError", "possession": "No Change", "possession_team": "NullError", "play_pattern": "NullError", "team": "NullError", "duration": "No Change", "tactics": "NullError", "related_events": "No Change", "player": "NullError", "position": "NullError", "location": -597, "pass": -1072, "carry": -533, "under_pressure": "No Change", "ball_receipt": "No Change", "duel": 37, "counterpress": "No Change", "interception": "No Change", "clearance": "No Change", "shot": -1, "goalkeeper": 14, "off_camera": "No Change", "dribble": "No Change", "foul_committed": "No Change", "foul_won": "No Change", "out": "No Change", "ball_recovery": "No Change", "miscontrol": "No Change", "substitution": "No Change", "injury_stoppage": "No Change"}
    nulls
    nullgoods = []
    for col in nulls:
        if nulls[col] != "No Change":
            nullgoods.append(col)
    nullgoods

    # %%
    """
    Soccermap: "startlocation": ["start_x_a0", "start_y_a0"],
                    "endlocation": ["end_x_a0", "end_y_a0"],
                    "freeze_frame_360": ["freeze_frame_360_a0"], - can i just get away with labeling passes??? maybe???
    xg_boost: |> ["freeze_frame_360", "start_x", "start_y"] 
    features:
    freeze_frame_360 -> "freeze_frame_360"
    start_x, start_y -> "location"
    end_x, end_y -> "end_location" in "pass"
    speed

    labels:
    success / fails -> "outcome" in "pass"(DNE when successful)
    xg -> "statsbomb_xg" in "shot"
    recipient -> "end_location" in "pass" and "freeze_frame_360"

    pass_options:
        - origin_x 
        - origin_y
        - destination_x 
        - destination_y
        - distance 
        - angle 
        - origin_angle_to_goal
        - destination_angle_to_goal
        - destination_distance_defender
        - pass_distance_defender
    """

    # %%
    statsbomb_event.columns
    done = ["id", "period", "minute", "second", "possession", "possession_team", "play_pattern", "timestamp", "index", 'type',
            "team", "duration", "tactics", "related_events", "player", "position", "location", "pass", "carry", "under_pressure", 
            "ball_receipt", "counterpress", "interception", "clearance", "duel", "shot", "goalkeeper"]
    #To look at: pass(this MUST be perfect!!!!)
    #Need to insert starting eleven
    todo = statsbomb_event.columns[~statsbomb_event.columns.isin(done)]
    todo

    # %%
    event["NeedsFlip_x"]

    # %%
    s11 = load_players(players_p, False)
    s11 = s11[s11["Starting"] == "true"]
    s11["position_dict"] = s11["PersonId"].map(players).map(position_map)
    s11["name"] = s11["FirstName"] + " " + s11["LastName"]
    def convert_to_dict(row):
        return {"id":row["PersonId"], "name":row["name"]} 
    s11["name_dict"] = s11.apply(convert_to_dict, axis = 1)
    s11["ShirtNumber"] = s11["ShirtNumber"].astype(int)
    def consolidate_dict(row):
        return {"player":row["name_dict"], "position":row["position_dict"], "jersey_number":row["ShirtNumber"]}
    s11["player_dict"] = s11.apply(consolidate_dict, axis = 1)
    teams = s11["TeamId"].unique()
    s11["team_dict"] = s11.apply(lambda d: {"id":d["TeamId"], "name":d["TeamName"]}, axis = 1)
    team1 = teams[0]
    team2 = teams[1]
    team1_players = list(s11[s11["TeamId"] == team1]["player_dict"])
    team2_players = list(s11[s11["TeamId"] == team2]["player_dict"])
    team_dict1 = s11[s11["TeamId"] == team1]["team_dict"].reset_index(drop=True).loc[0]
    team_dict2 = s11[s11["TeamId"] == team2]["team_dict"].reset_index(drop=True).loc[0]
    team1_formation = s11[s11["TeamId"] == team1]["Formation"].reset_index(drop=True).loc[0].replace("-", "").split(" ")[0]
    team2_formation = s11[s11["TeamId"] == team2]["Formation"].reset_index(drop=True).loc[0].replace("-", "").split(" ")[0]
    if team_dict1["id"] == "team1":
        team1_dict_t = team_dict1#this is a really stupid way to do this! revisit please!
        team2_dict_t = team_dict2
    else:
        team1_dict_t = team_dict2
        team2_dict_t = team_dict1

    firsttwo = pd.DataFrame([converted_event.loc[0], converted_event.loc[0]])
    relevant_cols = ["id", "index", "period", "timestamp", "minute", "second", "type", "possession",
                    "possession_team", "play_pattern", "team", "duration", "tactics"]
    firsttwo[(firsttwo.columns.difference(relevant_cols))] = None
    firsttwo
    #type: {'id': 35, 'name': 'Starting XI'}
    #team: {'id': 914, 'name': 'Italy'}
    firsttwo["type"] = [{'id': 35, 'name': 'Starting XI'}] * 2
    firsttwo["play_pattern"] = [{'id': 1, 'name': 'Regular Play'}] * 2
    firsttwo["id"] = [0,1]
    firsttwo["possession"] = 1
    firsttwo["team"] = [team_dict1, team_dict2]
    team1_tactics = {"formation": int(team1_formation), "lineup": team1_players}
    team2_tactics = {"formation": int(team2_formation), "lineup": team2_players}
    firsttwo["tactics"] = [team1_tactics, team2_tactics]
    firsttwo = firsttwo.reset_index(drop = True)

    # %%
    converted_s11 = pd.concat([firsttwo, converted_event]).reset_index(drop=True)
    converted_s11["index"] = converted_s11.index + 1
    converted_s11["timestamp"] = converted_s11["timestamp"].dt.time.astype(str)
    converted_s11["timestamp"] = np.where(converted_s11["timestamp"].str.contains("\."),converted_s11["timestamp"], converted_s11["timestamp"] + ".000")

    # %%
    def replace_if_nan(d):
        if isinstance(d, dict):
            if "end_location" not in d.keys():
                return None
        return d
    converted_s11["pass"] = converted_s11['pass'].apply(replace_if_nan)
    converted_s11["type"] = np.where(~pd.isnull(converted_s11["pass"]), {'id': 30, 'name': 'Pass'}, converted_s11["type"])
    converted_s11["type"] = np.where(~pd.isnull(converted_s11["shot"]), {'id': 16, 'name': 'Shot'}, converted_s11["type"])

    # %%
    together[together["EVENT_ID"] == 18464700001782.0]

    # %%
    converted_s11[converted_s11["id"] == 18464700001782.0]

    # %%
    shot_condition = (converted_s11['type'] == {"id": 16, "name": "Shot"}) & (converted_s11['shot'].isnull())#this is really lazy programming
    pass_condition = (converted_s11['type'] == {"id": 30,"name": "Pass"}) & (converted_s11['pass'].isnull())
    converted_s11 = converted_s11[~shot_condition]#18464700001782
    converted_s11 = converted_s11[~pass_condition]

    # %%
    converted_s11["id"].astype(int)

    # %%

    # Use .apply() to compare individual dictionary fields in the 'type' column
    #mask = (converted_s11["shot"].isna()) & (converted_s11["type"].apply(lambda x: x['id'] == 16 and x['name'] == 'Shot'))

    # Drop the rows where the condition is True
    #converted_s11 = converted_s11.drop(converted_s11[mask].index)
    

    # %%
    import json
    def convert_to_json(df, path):
            json_str = df.to_json(orient='records')
            json_data = json.loads(json_str)
            cleaned_data = json_data
            with open(path, 'w') as json_file:
                    json.dump(cleaned_data, json_file, indent=2)
    #converted_s11[converted_s11["id"] == 18468800000817]#["shot"]
    #18468800000814
    #converted_s11[converted_s11["id"] == 18468800000814]#["shot"]
    #converted_s11[converted_s11["pass"].notna()]
    convert_to_json(converted_s11, event_outpath)
    #print(tracking_outputpath)
    convert_to_json(tracking_converted, tracking_outpath)


