"""Implements the component for computing the expected value of a pass with added packing"""

import copy
from typing import Any, Callable, Dict, List, Optional, Union

import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xgboost as xgb
from rich.progress import track
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from socceraction import xthreat
from torch.utils.data import DataLoader, Subset

from unxpass.components.soccermap import SoccerMap, pixel
from unxpass.config import logger as log
from unxpass.datasets import PassesDataset
from unxpass.features import simulate_features

from unxpass.components.base import UnxpassComponent, UnxPassPytorchComponent, UnxPassXGBoostComponent


class PassValueComponent(UnxpassComponent):
    """The pass value component.

    From any given game situation where a player controls the ball, the model
    estimates the probability of scoring or conceding a goal following a pass.
    """

    component_name = "pass_value"

    def _get_metrics(self, y, y_hat):
        if "scores" in y:
            _y = y[["scores"]]
        elif "concedes" in y:
            _y = y[["concedes"]]
        else:
            return {}
        return {
            "brier": brier_score_loss(_y, y_hat),
            "log_loss": log_loss(_y, y_hat),
            "roc_auc": roc_auc_score(_y, y_hat),
        }

class PytorchSoccerMapModel(pl.LightningModule):
    """A pass value model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-6,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels= 13)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        # x = 2 * self.sigmoid(x) - 1  # scale to [-1, 1] to train on xG difference
        x = self.sigmoid(x)
        return x

    def step(self, batch: Any):
        x, mask, y = batch
        surface = self.forward(x)
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104), label="scores_xg"):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self.label = label

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        start_x, start_y, end_x, end_y = (
            sample["start_x_a0"],
            sample["start_y_a0"],
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])
        target = None
        if self.label in sample:
            target = float(sample[self.label])

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])
        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(
            -1, 2
        )
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # Output
        matrix = np.zeros((13, self.y_bins, self.x_bins))

        # CH 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        # CH 3: Distance to ball
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        
        players_att_vx = frame.loc[~frame.actor & frame.teammate, "x_velo"].values  # shape: (n_att_players,)
        players_att_vy = frame.loc[~frame.actor & frame.teammate, "y_velo"].values  # shape: (n_att_players,)

        # Similarly for defending players, using the correct condition (assuming defenders are non-teammates)
        players_def_vx = frame.loc[~frame.teammate, "x_velo"].values  # shape: (n_def_players,)
        players_def_vy = frame.loc[~frame.teammate, "y_velo"].values 
        # Channels 8 and 9, attacking team velocities 
        matrix[7, y_bin_att, x_bin_att] = players_att_vx
        matrix[8, y_bin_att, x_bin_att] = players_att_vy

        # Channel 10 & 11: Defending player velocities (x and y)
        matrix[9, y_bin_def, x_bin_def] = players_def_vx
        matrix[10, y_bin_def, x_bin_def] = players_def_vy

        # CH 12: Number of possession team’s players between the ball and every other location.
        dist_att_goal = matrix[0, :, :] * matrix[3, :, :]
        dist_att_goal[dist_att_goal == 0] = np.nan
        dist_ball_goal = matrix[3, y0_ball, x0_ball]
        player_in_front_of_ball = dist_att_goal <= dist_ball_goal
        
        outplayed1 = lambda x: np.sum(
            player_in_front_of_ball & (x <= dist_ball_goal) & (dist_att_goal >= x)
        )
        matrix[11, :, :] = np.vectorize(outplayed1)(matrix[3, :, :])

        # CH 13: Number of opponent team’s players between the ball and every other location.
        dist_def_goal = matrix[1, :, :] * matrix[3, :, :]
        dist_def_goal[dist_def_goal == 0] = np.nan
        dist_ball_goal = matrix[3, y0_ball, x0_ball]
        player_in_front_of_ball = dist_def_goal <= dist_ball_goal
        
        outplayed2 = lambda x: np.sum(
            player_in_front_of_ball & (x <= dist_ball_goal) & (dist_def_goal >= x)
        )
        matrix[12, :, :] = np.vectorize(outplayed2)(matrix[3, :, :])

        # Mask
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1

        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            None,
        )


class SoccerMapComponent(PassValueComponent, UnxPassPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, model: PytorchSoccerMapModel, offensive=True, success=True):
        self.label = "scores" if offensive else "concedes"
        self.success = success
        super().__init__(
            model=model,
            features={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "freeze_frame_360": ["freeze_frame_360_a0"],
            },
            label=[self.label, f"{self.label}_xg", "success"],
            transform=ToSoccerMapTensor(dim=(68, 104), label=f"{self.label}_xg"),
        )

    def initialize_dataset(self, dataset: Callable) -> PassesDataset:
        data = dataset(xfns=self.features, yfns=self.label, transform=self.transform)
        t = data.labels.reset_index()
        return data
        # return Subset(data, t.index[t.success == self.success].values)

    def test(self, dataset, config={}) -> Dict:
        train_cfg = config.get("train_cfg", {})

        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=train_cfg.get("batch_size", 64),
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=train_cfg.get("pin_memory", False),
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds, all_targets = [], []
        for batch in track(dataloader):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
            all_targets.append(y)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        all_targets = torch.cat(all_targets, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        return self._get_metrics(data.labels[self.label], all_preds)
        return self._get_metrics(data.labels[[self.label]], all_preds)

    @classmethod
    def load(cls, model_uri, **kwargs):
        loaded_model = mlflow.pytorch.load_model(model_uri + "/model", **kwargs)
        return cls(loaded_model)