import subprocess
import time
import warnings
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

import lightning.pytorch.loggers as pl_loggers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning, Callback
from sklearn.metrics import r2_score
from torch import nn

import wandb
from kirby.data import Collate, Dataset
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
        return [self.optimizer], [
            {"scheduler": self.scheduler, "interval": "step"}
        ]

    def setup(self, stage=None):
        # Make specific loggers available.
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                self.wandb = logger.experiment
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb = logger.experiment

    def on_train_start(self):
        # Log the output of `cat /proc/meminfo` using a shell script.
        try:
            # Execute the command and capture its output
            result = subprocess.run(
                ["cat", "/proc/meminfo"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            result = ""

        # Log the output
        console.info(f"Memory info: \n{result}")

    def on_train_epoch_start(self):
        self.epoch_time = time.time()

    def training_step(self, data, data_idx):
        output, loss, taskwise_loss = self.model(**data, compute_loss=True)

        # Compute the mean and std of the output.
        for name, val in output.items():
            self.log(f"outputs/mean_{name}", val.mean(), prog_bar=False)
            self.log(f"outputs/std_{name}", val.std(), prog_bar=False)
            self.log(
                f"targets/mean_{name}",
                data["output_values"][name].to(torch.float).mean(),
                prog_bar=False,
            )
            self.log(
                f"targets/std_{name}",
                data["output_values"][name].to(torch.float).std(),
                prog_bar=False,
            )

        if "spike_ids" in data:
            s = data["spike_ids"].to(torch.float)
            self.log("inputs/mean_spike_ids", s.mean(), prog_bar=False)
            self.log("inputs/std_spike_ids", s.std(), prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        for tag, value in self.model.named_parameters():
            self.log(f"vals/mean_{tag}", value.cpu().mean(), sync_dist=True)
            self.log(f"vals/std_{tag}", value.cpu().std(), sync_dist=True)
            if value.grad is not None:
                self.log(
                    f"grads/mean_{tag}",
                    value.grad.cpu().mean(),
                    sync_dist=True,
                )

        self.log("epoch_time", time.time() - self.epoch_time)

    def validation_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass


class CustomValidator(Callback):
    def __init__(self, validation_dataset: Dataset, collator: Collate):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.collator = collator

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: TrainWrapper
    ):
        # Perform custom validation here.
        pred = defaultdict(list)
        gt = defaultdict(list)  # Ground truth
        behavior_type = defaultdict(list)
        description = defaultdict(list)

        """We validate against behaviour using R2, so we must accumulate over batches."""
        for i in tqdm(range(trainer.local_rank, len(self.validation_dataset), trainer.world_size),
                      desc=f"Val @ Epoch {trainer.current_epoch}",
                      disable=(trainer.local_rank != 0)):
            # Samples are cyclically distributed across processes
            data = self.validation_dataset[i]
            session_id = data.session
            gt_, pred_ = stitched_prediction(
                data, self.collator, pl_module.model, pl_module.device
            )
            behavior_type_ = (
                data.behavior.type if hasattr(data.behavior, "type") else None
            )

            for task_type in pred_.keys():
                gt[(session_id, task_type)].append(gt_[task_type])
                pred[(session_id, task_type)].append(pred_[task_type])
                behavior_type[(session_id, task_type)].append(behavior_type_)
                description[(session_id, task_type)].append(data.description)

        def gather_concat_dict(obj):
            """Gather and concatenate dictionary-of-list objects onto
            the rank=0 process
            """
            gathered_objlist = None
            if trainer.local_rank == 0:
                gathered_objlist = [None] * trainer.world_size

            dist.gather_object(obj, gathered_objlist, 0)

            # Concatenate all lists
            gathered_obj = None
            if trainer.local_rank == 0:
                gathered_obj = defaultdict(list)
                for i, objlist in enumerate(gathered_objlist):
                    for k in objlist:
                        gathered_obj[k] += objlist[k]

            dist.barrier()
            return gathered_obj

        # Gather
        if trainer.world_size > 1:
            gt = gather_concat_dict(gt)
            pred = gather_concat_dict(pred)
            behavior_type = gather_concat_dict(behavior_type)
            description = gather_concat_dict(description)

        if trainer.local_rank != 0:
            return

        r2 = defaultdict(dict)
        for session_id, task_type in gt.keys():
            # Resolve the right metric for the session.
            gt_ = torch.cat(gt[(session_id, task_type)], dim=0).detach().cpu()
            pred_ = (
                torch.cat(pred[(session_id, task_type)], dim=0).detach().cpu()
            )
            # TODO: reintegrate this functionality into the new metric system.
            # behavior_type_ = torch.cat(behavior_type[(session_id, task_type)], dim=0)
            # .detach().cpu()

            desc = description[(session_id, task_type)][-1].metrics
            desc = [x for x in desc if x.output_key == task_type]
            if not desc:
                raise ValueError(
                    f"Cannot find description for {session_id}, {str(task_type)}"
                )
            if len(desc) > 1:
                raise ValueError(
                    f"Found multiple descriptions for {session_id}, {str(task_type)}"
                )

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
                metric, task_spec.type, pred_, gt_, 1.0
            )

            r2[session_id][
                f"{metric}_{str(task_type.lower())}"
            ] = the_metric.item()

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

        pl_module.log_dict(values)
        console.info(f"Logged {len(values)} validation metrics.")

        r2 = pd.DataFrame.from_dict(r2).T
        if pl_module.tb is not None:
            pl_module.tb.add_text("val_r2", r2.to_markdown())
        if pl_module.wandb is not None:
            pl_module.wandb.log({"val_r2": wandb.Table(dataframe=r2)})

        return r2


class UnfreezeAtEpoch(Callback):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self._unfreeze_at_epoch:
            console.info(
                f"Reached epoch {trainer.current_epoch}, unfreezing entire model"
            )
            for module in pl_module.model.children():
                for param in module.parameters():
                    param.requires_grad = True
