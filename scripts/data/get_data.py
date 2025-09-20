import config
from skillcorner.client import SkillcornerClient
import pandas as pd
import io
from pathlib import Path
from tqdm import tqdm

sc_username = config.sc_username
sc_password = config.sc_password
current_file_directory = Path(__file__).parent

rdf_path = Path("../../../rdf/sp161/shared/asi_gk_pos/data")

client = SkillcornerClient(username = sc_username, password = sc_password)
matches_meta = []
matches = [match['id'] for match in client.get_matches()]
errors = []
for match in tqdm(matches):
    try:
        match_tracking = client.get_match_tracking_data(match_id = match)
        match_meta = client.get_match(match_id = match)
        match_events = client.get_dynamic_events(match_id = match)

        matches_meta.append(match_meta)
        match_tracking = pd.DataFrame(match_tracking)
        match_events = pd.read_csv(io.BytesIO(match_events))

        match_tracking.to_csv(rdf_path / "tracking" / f"{match}_tracking.csv", index = False)
        match_events.to_csv(rdf_path / "event" / f"{match}_events.csv", index = False)
    except Exception as e:
        print(f"Error for match {match}: {e}")
        errors.append(match)
pd.DataFrame(matches_meta).to_csv(rdf_path / "matches_meta.csv", index = False)
print(errors)

"""
errors = [2016424, 2016425, 2015190, 2013662, 2013663, 2013665, 
2013070, 2012511, 2009510, 2009421, 2009422, 2009511, 2009513, 
2008430, 2007936, 2007466, 2006146, 2005631, 2005632, 2005633, 
2003683, 2003684, 2003685, 2001858, 1987409, 1989332, 1975031, 
1914360, 1904687, 1863386, 1863387, 1862012, 1850168, 1850169, 
1847609, 1835426, 1836363, 1805093, 1805096, 1751878, 1752982, 
1735628, 1731383, 1732596, 1728459, 1700744, 1668542, 1670500, 
1595704, 1580755, 1581984, 1571937, 1571938, 1552777, 1508311, 1461421]
"""