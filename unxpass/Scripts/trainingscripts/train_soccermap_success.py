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
from unxpass.components.withSpeeds import pass_success_speeds
from unxpass.components.utils import log_model, load_model
from unxpass import __main__, utils
from omegaconf import DictConfig, OmegaConf
STORES_FP = Path("../stores")
db_path = "../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/buli_all.sql"
db = SQLiteDatabase(db_path)

config_fp = "/home/lz80/un-xPass/config/"

overrides = ["experiment=pass_success/soccermap"]
cfg = __main__.parse_config(config_path=config_fp, overrides = overrides)

pass_success_model = pass_success_speeds.SoccerMapComponent(model = pass_success_speeds.PytorchSoccerMapModel())

train_cfg = OmegaConf.to_object(cfg.get("train_cfg", DictConfig({})))
utils.instantiate_callbacks(train_cfg)
utils.instantiate_loggers(train_cfg)
custom_path = "../../../../rdf/sp161/shared/soccer-decision-making/Bundesliga/features/features_filtered"
dataset_train = partial(PassesDataset, path=custom_path)
logger.info("Starting training!")
mlflow.set_experiment(experiment_name="pass_success/soccermap")
with mlflow.start_run() as run:
    # Log config
    with tempfile.TemporaryDirectory() as tmpdirname:
        fp = Path(tmpdirname)
        OmegaConf.save(config=cfg, f=fp / "config.yaml")
        mlflow.log_artifact(str(fp / "config.yaml"))
    pass_success_model.train(dataset_train, optimized_metric=cfg.get("optimized_metric"), **train_cfg)
    log_model(pass_success_model.model, "model")
    run_id = run.info.run_id
    print(f"Pass Success Model saved with run_id: {run_id}")
logger.info("✅ Finished training. Model saved with ID %s", run.info.run_id)
