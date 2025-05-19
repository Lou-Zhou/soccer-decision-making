import xml.etree.ElementTree as ET
import pandas as pd
#Functions loading buli data to df
def load_tracking(path):
    tree = ET.parse(path) 
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
                    #'D': elem.attrib.get('D'),
                    #'S': elem.attrib.get('S'),
                    #'A': elem.attrib.get('A'),
                    #'M': elem.attrib.get('M')
            }
            data.append(entry)

    df = pd.DataFrame(data)
    return df
    

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
def load_event(path):
        tree = ET.parse(path) 
        root = tree.getroot()
        data = []
        def parse_event(event_element, parent_data):
            for child in event_element:
                child_data = {**parent_data, **child.attrib}
                if list(child):
                    parse_event(child, child_data)
                else:
                    data.append(child_data)
        for event in root.findall('Event'):
            event_data = event.attrib 
            for child in event:
                if child.tag == "Play":
                    for subchild in child:
                        event_data["EventType"] = subchild.tag
                else:
                    event_data["EventType"] = child.tag
            parse_event(event, event_data)
        df = pd.DataFrame(data)
        return df
def load_csv_event(path):
    events = pd.read_csv(path ,sep = ';', encoding='latin-1', on_bad_lines='skip')
    return events