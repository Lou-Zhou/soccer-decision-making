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
from unxpass.components.withSpeeds import pass_selection_speeds
from unxpass.components.utils import log_model, load_model
from unxpass import __main__, utils
from omegaconf import DictConfig, OmegaConf
import hydra

from sdm import path_data, path_repo

# Handle file paths ----

path_db = path_data + "/Bundesliga/buli_all.sql"
path_config = path_repo + "/config/"
STORES_FP = Path("../stores")


db = SQLiteDatabase(path_db)

overrides = ["experiment=pass_selection/soccermap"]
cfg = __main__.parse_config(config_path=path_config, overrides = overrides)
train_cfg = OmegaConf.to_object(cfg.get("train_cfg", DictConfig({})))
utils.instantiate_callbacks(train_cfg)
utils.instantiate_loggers(train_cfg)

custom_path = path_data + "/Bundesliga/features/features_angle"
dataset_train = partial(PassesDataset, path=custom_path)
logger.info("Starting training!")
pass_selection_model = hydra.utils.instantiate(cfg["component"], _convert_="partial")
mlflow.set_experiment(experiment_name="pass_selection/soccermap")
with mlflow.start_run() as run:
    with tempfile.TemporaryDirectory() as tmpdirname:
        fp = Path(tmpdirname)
        OmegaConf.save(config=cfg, f=fp / "config.yaml")
        mlflow.log_artifact(str(fp / "config.yaml"))
    pass_selection_model.train(dataset_train, optimized_metric=cfg.get("optimized_metric"), **train_cfg)
    mlflow.pytorch.log_model(pass_selection_model.model, "component")
    run_id = run.info.run_id
    print(f"Pass Selection Model saved with run_id: {run_id}")
logger.info("✅ Finished training. Model saved with ID %s", run.info.run_id)
