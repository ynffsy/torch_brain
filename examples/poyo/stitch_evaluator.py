from collections import defaultdict
import logging

import pandas as pd
from rich import print as rprint
import torch
import lightning as L
import wandb

from torch_brain.utils.stitcher import stitch


class StitchEvaluator(L.Callback):
    def __init__(self, metric_fn, quiet=False):
        self.metric_fn = metric_fn
        self.quiet = quiet

    def on_validation_epoch_start(self, trainer, pl_module):
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

            self.cache[session_id]["pred"].append(pred.detach().cpu())
            self.cache[session_id]["target"].append(target.detach().cpu())
            self.cache[session_id]["timestamps"].append(timestamps.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
        metrics = {}
        for session_id in self.cache.keys():
            cache = self.cache[session_id]
            pred = torch.cat(cache["pred"])
            target = torch.cat(cache["target"])
            timestamps = torch.cat(cache["timestamps"])

            stitched_pred = stitch(timestamps, pred)
            stitched_target = stitch(timestamps, target)

            metrics[session_id] = self.metric_fn(stitched_pred, stitched_target)

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
