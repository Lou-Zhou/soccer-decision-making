#Helper function which simplifies the output of evaluation and selection criterions to aggregate to one single number
import pandas as pd
import numpy as np

def group_agg(input_dir, output_dir):
    samples = pd.read_csv(input_dir)
    samples["split"] = samples['original_event_id'].str.split('-')
    samples['home_pass'] = np.where(samples['split'].str.len() == 6, 
                                samples['original_event_id'].str.rsplit('-', 1).str[0], 
                                samples['original_event_id'])
    result = samples.groupby('home_pass').agg({
        'game_id': 'first',           # average for this column
        'action_id': 'first',         # first value for this column
        'start_x': 'first',
        'start_y': 'first',
        'selection_criterion':'mean'
    }).assign(numFrames=samples.groupby('home_pass').size())
    result.to_csv(output_dir)
input_dir = '/home/lz80/un-xPass/unxpass/eurotest_womens_criterias_reception.csv'
output_dir = 'receiptions_grouped.csv'
group_agg(input_dir, output_dir)