This repository is code originally from [un-xPass: Measuring Soccer Player’s Creativity](https://github.com/ML-KULeuven/un-xPass) adapted to quantify the quality of decision-making in soccer for Hawkeye and SkillCorner data by training various Soccermap Models to analyze the success, selection, and value of passes to potential recipients.

## Installation

Getting Started, requiring [Poetry](https://python-poetry.org/):

```sh
GNUTLS_CPUID_OVERRIDE=0x1 git clone https://github.com/Lou-Zhou/soccer-decision-making.git #GNUTLS_CPUID_OVERRIDE=0x1 only included due to some errors with cloning / https connections
sudo apt update
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global #install pipx, dependency for poetry
pipx install poetry #installing poetry, dependency for un-x
python3 -m venv .venv
cd soccer-decision-making
```

Additionally, update sdm/config.ini with the appropriate paths on your machine.

### Setup

You need to run the following lines every time to set up the virtual environment with the appropriate modules installed.

```sh
source .venv/bin/activate
```

## Generating Parquet Features

Since data is formatted as a directory of parquet files, various python scripts can be used to get these features from tracking and event data for all three data formats for the soccermap models:

1. Sportec Data - scripts/features/getBuliFeats.py
2. Hawkeye Data - scripts/features/getHawkeyeFeats.py
3. SkillCorner Data - scripts/features/getSkillCornerFeatures.py

For the Hawkeye features, since we are looking at times surrounding an event as well, model outputs will be specified in a (game_id, action_id-frame_index) way, where frame_index describes the number of frames from the original reception. In addition, editFeatures.py should be run to edit features as needed(e.g. getting only the successful passes for the value model)

## Training Models

```sh
python3 scripts/training/train_soccermap_selection.py     # OR *_success.py OR *_selection.py
```

The config files (for hyperparameters) are, for example, `config/experiment/pass_selection/soccermap.yaml`.

In `train_soccermap_value.py`, it is important to change the `experiment` variable (e.g. "pass_value/soccermap_offensive_completed") so that it matches the model desired.

The paths to the configurations in all these scripts must be set manually as these paths must be absolute.

These scripts will output both the last trained model (in run_id form) as well as the model with the smallest lost(in checkpoint form). Using unxpass/Scripts/helperScripts/checkpntToModel.py, we can then turn these checkpoints into run_ids so we can standardize model storage. The paths to these checkpoints can be found from the code outputs or in trainingscripts/lightning_logs/version_x/checkpoints.

## Visualizing Results

Here is an example of how to visualize the first 200 passes of one game for a trained selection model, predicting on Hawk-Eye data (takes about a minute):

```sh
import sdm.visualization

sdm.visualization.plot_model_outputs(
    component = "selection",
    run_id = "cb051b26ef7640f9834e360fb3ca0c1b",
    path_feature = "Hawkeye/Hawkeye_Features/sequences",
    game_id = 3835320,
    show_pass = False
)
```

There is some additional useful code in scripts/vizualization:

1. animatePlays.py - animates given sequence of frames from the tracking data
2. animateSurfaces.py - animates model surfaces over sequences of frames, useful for Hawkeye data
3. compareFeatures.py - debugging tool to compare one set of features with another
4. plotSpeeds.py - plots the speed of a player over the course of a game (used to check speed smoothing values)
5. visualizeFeatures.py - generates visualizations of game states from tracking data
6. visualizeModelOutput.py - generates singular visualizatios of model surface 

## Getting Results

Using resultGenerators/getResults.py, we can then generate two csvs: 

1. allModelOutputs.csv - describes the model outputs for every frame evaluated in the data
2. allModelOutputsAggregated.csv - describes the model outputs aggregated for every event in the data

## Hyperparameter Tuning

Hyperparameter tuning can be done using run_experiment.py. For example, the following line performs hyperparameter tuning for the success probability model:

```sh
python3 run_experiment.py experiment="pass_success/soccermap" hparams_search="soccermap_optuna"
```

In `config/hparams_search/soccermap_optuna.yaml`, the search over the learning rates and batch sizes can be changed as desired. This script will generate a path to a checkpoint which can then be turned into a run_id with, for example:
```sh
python3 scripts/helper/checkpntToModel.py epoch_014.ckpt
```
This script creates corresponding model files in `stores/model/`.

It should be noted that the un-xPass paper uses the following hyperparameter methods, which are slightly different from the current hyperparameter setup:

> We perform a grid search on the learning rate (1𝑒−3, 1𝑒−4, 1𝑒−5, 1𝑒−6), and batch size parameters (16, 32, 64). We use early stopping with patience set to 10 epochs and a delta of 1𝑒−3 for the pass success probability model, and 1𝑒−5 for the pass selection and pass value models
