from collections import defaultdict
from typing import Optional
import warnings

import lightning.pytorch.loggers as pl_loggers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import r2_score
from torch import nn
from kirby.data import Dataset

import wandb
from kirby.data import Collate
from kirby.data.stitcher import stitched_prediction
from kirby.models.perceiver_rotary import compute_metric
from kirby.tasks.reaching import REACHING
from kirby.taxonomy.taxonomy import Task
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
        self.tb = None
        self.wandb = None

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def setup(self, stage=None):
        # Make specific loggers available.
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb = logger.experiment
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb = logger.experiment

    def training_step(self, data, data_idx):
        output, loss, taskwise_loss = self.model(
            **data, compute_loss=True
        )

        # Compute the mean and std of the output.
        for name, val in output.items():
            self.log(f"outputs/mean_{name}", val.mean(), prog_bar=False)
            self.log(f"outputs/std_{name}", val.std(), prog_bar=False)
            self.log(f"targets/mean_{name}", data['output_values'][name].to(torch.float).mean(), prog_bar=False)
            self.log(f"targets/std_{name}", data['output_values'][name].to(torch.float).std(), prog_bar=False)

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
    def __init__(self, validation_dataset: Dataset, collator: Collate):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.collator = collator

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: TrainWrapper):
        # Perform custom validation here.
        pred = defaultdict(list)
        gt = defaultdict(list)  # Ground truth
        behavior_type = defaultdict(list)
        description = defaultdict(list)

        """We validate against behaviour using R2, so we must accumulate over batches."""
        for data in self.validation_dataset:
            session_id = data.session
            gt_, pred_ = stitched_prediction(
                data, self.collator, pl_module.model, pl_module.device
            )
            behavior_type_ = (
                data.behavior.type
                if hasattr(data.behavior, "type")
                else data.behavior.behavior_type
            )

            for task_type in pred_.keys():
                gt[(session_id, task_type)].append(gt_[task_type])
                pred[(session_id, task_type)].append(pred_[task_type])
                behavior_type[(session_id, task_type)].append(behavior_type_)
                description[(session_id, task_type)].append(data.description)
            

        r2 = defaultdict(dict)
        for session_id, task_type in gt.keys():
            # Resolve the right metric for the session.
            gt_ = torch.cat(gt[(session_id, task_type)], dim=0).detach().cpu()
            pred_ = torch.cat(pred[(session_id, task_type)], dim=0).detach().cpu()
            behavior_type_ = torch.cat(behavior_type[(session_id, task_type)], dim=0).detach().cpu()

            desc = description[(session_id, task_type)][-1].metrics
            desc = [x for x in desc if x.output_key == task_type]
            if not desc:
                raise ValueError(f"Cannot find description for {session_id}, {str(task_type)}")
            if len(desc) > 1:
                raise ValueError(f"Found multiple descriptions for {session_id}, {str(task_type)}")
            
            desc = desc[0]
            
            metric = None
            if hasattr(desc, "metric"):
                metric = desc.metric
            else:
                # Get it from the model spec.
                metric = pl_module.model.readout.task_specs[task_type].loss_fn

            task_spec = pl_module.model.readout.task_specs[task_type]

            # Resolve the appropriate loss function.
            the_metric = compute_metric(
                metric,
                task_spec.type, 
                pred_, 
                gt_, 
                1.0)
            
            r2[session_id][f"{metric}_{str(task_type.lower())}"] = the_metric.item()


            # TODO: reintegrate this functionality into the new metric system.
            """
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
                warnings.warn(
                    f"Cannot infer behavior type from session {session_id}",
                    RuntimeWarning,
                )
            """
        
        # Fold the results into a single number.
        values = {}
        for key, item in r2.items():
            for key2, item2 in item.items():
                values[f"val/{key}_{key2}"] = item2

        pl_module.log_dict(values, sync_dist=True)
        console.info(f"Logged {len(values)} validation metrics.")

        r2 = pd.DataFrame.from_dict(r2).T
        if pl_module.tb is not None:
            pl_module.tb.add_text("val_r2", r2.to_markdown())
        if pl_module.wandb is not None:
            pl_module.wandb.log({"val_r2": wandb.Table(dataframe=r2)})

        return r2
