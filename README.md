This repository is code originally from [un-xPass: Measuring Soccer Player’s Creativity](https://github.com/ML-KULeuven/un-xPass) adapted to quantify the quality of decision-making in soccer for Hawkeye and SkillCorner data by training various Soccermap Models to analyze the success, selection, and value of passes to potential recipients.

## Installation

Getting Started, requiring [Poetry]{https://python-poetry.org/}:

```sh
$ git clone https://github.com/Lou-Zhou/soccer-decision-making.git
$ cd un-xPass
$ python3 -m venv .venv
$ source .venv/bin/activate
$ poetry install
```
## Generating Parquet Features

Since data formatted as a directory of parquet files, various python scripts can be used to get these features from tracking and event data for all three data formats for the soccermap models:
1. Sportec Data - unxpass/Scripts/featureGenerators/getBuliFeats.py
2. Hawkeye Data - unxpass/Scripts/featureGenerators/getHawkeyeFeats.py
3. SkillCorner Data - unxpass/Scripts/featureGenerators/getSkillCornerFeatures.py

For the Hawkeye features, since we are looking at times surrounding an event as well, model outputs will be specified in a (game_id, action_id-frame_index) way, where frame_index describes the number of frames from the original reception. In addition, editFeatures.py should be run to edit features as needed(e.g. getting only the successful passes for the value model or trimming superfluous frames as there may exist situations where another play happens 1 second after reception)

## Training Models

From these features, we can then train the three model components using the following scripts in unxpass/Scripts/trainingscripts. The models use the configurations as described in config/experiment and can be changed depending on need. These scripts will output both the last trained model(in run_id form) as well as the model with the smallest lost(in checkpoint form). Using unxpass/Scripts/helperScripts/checkpntToModel.py, we can then turn these checkpoints into run_ids so we can standardize model storage. 

For the four value models, it is important to change the experiment name(e.g. "pass_value/soccermap_offensive_completed") so that they match the model as desired.

In addition, the paths to the configurations in all these scripts must be set manually as these paths must be absolute.

## Visualizations

We can generate visualizations for both the plays and the model results using scripts found in Scripts/visualizationScripts:
1. getAnimations.py - generates animations for a Bundesliga play
2. getmodeloutput.py - generates model outputs for play(s)
3. plotSpeeds.py - plots the speed of a player over the course of a game(used to check speed smoothing values)
4. visualizeFeatures - plots the game states from the parquet files

## Getting Results

Using resultGenerators/getResults.py, we can then generate two csvs: 

1. allModelOutputs.csv - describes the model outputs for every frame evaluated in the data
2. allModelOutputsAggregated.csv - describes the model outputs aggregated for every event in the data

## Hyperparameter Tuning

Hyperparameter tuning can be done using run_experiment.py with the following command:

```
python3 run_experiment.py \
  experiment="experiment_name" \
  hparams_search="hparam_method" 
```

For the soccermap models, the experiment name will be of the form pass_(success/selection)/soccermap or pass_value/soccermap_{offensive/defensive}_{completed/failed} and the hyperparameter method for soccermap models will be "soccermap_optuna". In hparams_search/soccermap_optuna, the search over the learning rates and batch sizes can be changed as desired. This script will generate a path to a checkpoint which can then be turned into a run_id using helperScripts/checkpntToModel.py.

It should also be noted that all paths should be set so that they run properly(with proper rdf access) if the working directory is the folder that the script is in, besides the config paths, which must be asolute.