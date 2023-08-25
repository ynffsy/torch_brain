from collections import defaultdict
from typing import Optional

import lightning.pytorch.loggers as pl_loggers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

from kirby.data.stitcher import stitched_prediction
from kirby.tasks.reaching import REACHING
from kirby.utils import logging

console = logging(header="TRAIN WRAPPER", header_color="red")


class TrainWrapper(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def setup(self, stage=None):
        # Make specific loggers available.
        self.tb = None
        self.wandb = None
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb = logger.experiment
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb = logger.experiment

    def training_step(self, data, data_idx):
        output = self.model(
            data["spike_unit"],
            data["spike_timestamps"],
            data["spike_type"],
            data["input_mask"],
            data["latent_id"],
            data["latent_timestamps"],
            data["output_timestamps"],
            data["task_id"],
        )

        loss = F.mse_loss(output, data["output_values"], reduction="none")
        loss[~data["output_mask"]] = 0.0
        loss = loss * data["output_weight"].unsqueeze(-1)
        loss = loss.sum() / (data["output_weight"].sum())

        self.log("train_loss", loss, prog_bar=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        for tag, value in self.model.named_parameters():
            self.log(f"vals/mean_{tag}", value.cpu().mean(), sync_dist=True)
            self.log(f"vals/std_{tag}", value.cpu().std(), sync_dist=True)
            if value.grad is not None:
                self.log(f"grads/mean_{tag}", value.grad.cpu().mean(), sync_dist=True)
                self.log(f"grads/std_{tag}", value.grad.cpu().std(), sync_dist=True)

    def validation_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass


class CustomValidator(Callback):
    def __init__(self, validation_dataset: DataLoader, collator):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.collator = collator

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: TrainWrapper):
        # Perform custom validation here.
        pred = defaultdict(list)
        gt = defaultdict(list)  # Ground truth
        behavior_type = defaultdict(list)

        """We validate against behaviour using R2, so we must accumulate over batches."""
        for data in self.validation_dataset:
            session_id = data.session_id
            gt_, pred_ = stitched_prediction(
                data, pl_module.model, pl_module.device, self.collator
            )
            behavior_type_ = (
                data.behavior.type.numpy()
                if hasattr(data.behavior, "type")
                else data.behavior.behavior_type.numpy()
            )

            gt[session_id].append(gt_)
            pred[session_id].append(pred_)
            behavior_type[session_id].append(behavior_type_)

        r2 = defaultdict(
            lambda: dict(
                full=None, hold=None, reach=None, return_center=None, random=None
            )
        )
        for session_id in gt.keys():
            gt[session_id] = np.concatenate(gt[session_id], axis=0)
            pred[session_id] = np.concatenate(pred[session_id], axis=0)
            behavior_type[session_id] = np.concatenate(
                behavior_type[session_id], axis=0
            )

            r2[session_id]["full"] = r2_score(gt[session_id], pred[session_id])

            if "CO" in session_id:
                for behavior_type_id, name in [
                    (REACHING.CENTER_OUT_REACH, "reach"),
                    (REACHING.CENTER_OUT_RETURN, "return_center"),
                    (REACHING.CENTER_OUT_HOLD, "hold"),
                ]:
                    mask = behavior_type[session_id] == behavior_type_id
                    r2[session_id][name] = r2_score(
                        gt[session_id][mask], pred[session_id][mask]
                    )
            elif "RT" in session_id:
                for behavior_type_id, name in [
                    (REACHING.RANDOM, "random"),
                    (REACHING.HOLD, "hold"),
                ]:
                    mask = behavior_type[session_id] == behavior_type_id
                    r2[session_id][name] = r2_score(
                        gt[session_id][mask], pred[session_id][mask]
                    )
            else:
                raise NotImplementedError(
                    f"Cannot infer behavior type from session {session_id}"
                )
        r2 = pd.DataFrame.from_dict(r2).T
        if pl_module.tb is not None:
            pl_module.tb.add_text("val_r2", r2.to_markdown())
        if pl_module.wandb is not None:
            pl_module.wandb.log({"val_r2": wandb.Table(dataframe=r2)})
