
import os

from sdm import path_data
from sdm import featurization
from sdm import visualization

feature_dir = 'Hawkeye/Hawkeye_Features/2025-10-13'
output_dir = 'output/animations/2025-10-13'

featurization.get_features_hawkeye(
    output_dir = os.path.join(path_data, feature_dir),
    ball = True,
    frame_idxs = range(-125, 51),
    id = [
        '0291ae3b-9848-4f16-ab2e-2dc194ed8649',
        'a97f38e5-f4fd-4174-9b48-1436832ff654',
        '04420e29-c2df-4f19-8d56-ebdc8737dfd9',
    ],
)

visualization.generate_pass_surface_gifs(
    component = 'selection',
    run_id = 'cb051b26ef7640f9834e360fb3ca0c1b',
    path_feature = os.path.join(path_data, feature_dir),
    path_play = 'steffen/sequence_filtered.csv',
    path_output = output_dir,
)
