#trains success model from soccermap
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from unxpass.config import logger
import mlflow
import tempfile
from xgboost import XGBClassifier, XGBRanker
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass.components import pass_success
from unxpass.components.withSpeeds import pass_value_speeds
from unxpass.components.utils import log_model, load_model
from unxpass import __main__, utils
from omegaconf import DictConfig, OmegaConf
import hydra

from sdm import path_data, path_repo

# Handle file paths ----

path_config = path_repo + "/config/"


experiment = "pass_value/soccermap_offensive_failed"#experiment to be used, should be change depending on the model wanted
overrides = [f"experiment={experiment}"]
cfg = __main__.parse_config(config_path = path_config, overrides = overrides)
side = experiment.split("_")[2]
state = experiment.split("_")[3]
offensive = side == "offensive"
completed = state == "completed"

train_cfg = OmegaConf.to_object(cfg.get("train_cfg", DictConfig({})))
utils.instantiate_callbacks(train_cfg)
utils.instantiate_loggers(train_cfg)
custom_path = path_data + "/Bundesliga/features/features"
#features_fail_test_ogTest: original end locations with speed and impossible filtering
#custom_path = "/home/lz80/rdf/sp161/shared/soccer-decision-making/Bundesliga/features/oldFeatures/all_features_defl_fail"
if completed:
   custom_path = f"{custom_path}_success"
else:
   custom_path = f"{custom_path}_failed"
dataset_train = partial(PassesDataset, path=custom_path)
logger.info("Starting training!")
pass_value_model = hydra.utils.instantiate(cfg["component"], _convert_="partial")
mlflow.set_experiment(experiment_name=experiment)
with mlflow.start_run() as run:
    # Log config
    with tempfile.TemporaryDirectory() as tmpdirname:
        fp = Path(tmpdirname)
        OmegaConf.save(config=cfg, f=fp / "config.yaml")
        mlflow.log_artifact(str(fp / "config.yaml"))
    pass_value_model.train(dataset_train, optimized_metric=cfg.get("optimized_metric"), **train_cfg) 
    mlflow.pytorch.log_model(pass_value_model.model, "model")
    run_id = run.info.run_id
    print(f"Pass Value {side} {state} Model saved with run_id: {run_id}")
logger.info("✅ Finished training. Model saved with ID %s", run.info.run_id)
