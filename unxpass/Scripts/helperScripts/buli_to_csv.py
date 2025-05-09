import pandas as pd
from unxpass import load_xml
import numpy as np
from tqdm import tqdm
import os
def main():
    xmls = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/zipped_tracking/zip_output"
    outputDir = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/raw_data/tracking_csv/"
    for xml in tqdm(os.listdir(xmls)):
        xmlPath = f"{xmls}/{xml}"
        gameId = xml.split(".")[0]
        trackingData = load_xml.load_tracking(xmlPath)
        trackingData.to_csv(f"{outputDir}/{gameId}.csv")
if __name__ == "__main__": main()