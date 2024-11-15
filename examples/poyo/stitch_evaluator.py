from collections import defaultdict
import logging

import pandas as pd
from rich import print as rprint
import torch
import lightning as L
import torchmetrics
import wandb

from torch_brain.registry import ModalitySpec
from torch_brain.utils.stitcher import stitch


class StitchEvaluator(L.Callback):
    def __init__(self, session_ids: list, modality_spec: ModalitySpec, quiet=False):
        self.quiet = quiet

        self.metrics = {k: torchmetrics.R2Score(modality_spec.dim) for k in session_ids}

    def on_validation_epoch_start(self, trainer, pl_module):
        # Cache to store the predictions, targets, and timestamps for each
        # validation step. This will be coalesced at the end of the validation,
        # using the stitch function.
        self.cache = defaultdict(
            lambda: {
                "pred": [],
                "target": [],
                "timestamps": [],
            }
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update the cache with the predictions, targets, timestamps, and subtask index
        # TODO: Mask timestamps based on subtask_index
        batch_size = len(outputs)
        for i in range(batch_size):
            mask = batch["output_mask"][i]
            session_id = batch["session_id"][i]
            absolute_start = batch["absolute_start"][i]

            pred = outputs[i][mask]
            target = batch["target_values"][i][mask]
            timestamps = batch["output_timestamps"][i][mask] + absolute_start

            self.cache[session_id]["pred"].append(pred.detach())
            self.cache[session_id]["target"].append(target.detach())
            self.cache[session_id]["timestamps"].append(timestamps.detach())

    def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
        # compute metric for each session
        metrics = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self.cache[session_id]
            pred = torch.cat(cache["pred"])
            target = torch.cat(cache["target"])
            timestamps = torch.cat(cache["timestamps"])

            stitched_pred = stitch(timestamps, pred)
            stitched_target = stitch(timestamps, target)

            metric_fn.to(pl_module.device).update(stitched_pred, stitched_target)
            metrics[session_id] = metric_fn.compute()
            metric_fn.reset()

        # compute the average metric
        metrics[f"average_{prefix}_metric"] = torch.tensor(
            list(metrics.values())
        ).mean()

        # log the metrics
        self.log_dict(metrics)
        if not self.quiet:
            logging.info(f"Logged {len(metrics)} {prefix} metrics.")

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value.item()})

        metrics_df = pd.DataFrame(metrics_data)
        if not self.quiet:
            rprint(metrics_df)

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    logger.experiment.add_text(
                        f"{prefix}_metrics", metrics_df.to_markdown()
                    )
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_validation_epoch_start(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs, prefix="test")
