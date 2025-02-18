#Script to read buli data to gif
from unxpass import load_xml, visualizers_made
import matplotlib.animation as animation
events = load_xml.load_csv_event("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/KPI_Merged_all/KPI_MGD_DFL-MAT-J03YDU.csv")
tracking = load_xml.load_tracking("/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/zipped_tracking/zip_output/DFL-MAT-J03YDU.xml")
ani = visualizers_made.get_animation_from_raw(18453000000078, 40, events, tracking, range(14650, 14870), False)
writergif = animation.PillowWriter(fps=25)
ani.save('sample_goal.gif', writer=writergif)