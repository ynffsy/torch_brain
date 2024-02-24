"""A module that takes a long trial, chops it up into bite-sized pieces, processes it as
 usual, stitches it back together."""

import torch
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from lightning.pytorch.callbacks import Callback
import logging

from kirby.nn import compute_loss_or_metric
from kirby.taxonomy.taxonomy import Output, OutputType


class CustomValidator(Callback):
    def __init__(
        self,
        validation_loader,
    ):
        super().__init__()
        self.loader = validation_loader

    def on_validation_epoch_start(self, trainer, pl_module):
        session_timestamp = {}
        session_subtask_index = {}
        session_gt_output = {}
        session_pred_output = {}

        for batch in tqdm(
            self.loader,
            desc=f"Val @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            absolute_starts = batch.pop("absolute_start")  # (B,)
            session_ids = batch.pop("session_id")  # (B,)
            output_subtask_index = batch.pop("output_subtask_index")

            # move to gpu dict of dicts
            def move_to_gpu(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        move_to_gpu(v)
                    elif isinstance(v, torch.Tensor):
                        d[k] = v.to(pl_module.device)

            move_to_gpu(batch)

            # forward pass
            with torch.inference_mode():
                pred_output, loss, losses_taskwise = pl_module.model(**batch)

            # we need to get the timestamps, the ground truth values, the task ids as well
            # as the subtask ids. since the batch is padded and chained, this is a bit tricky
            # tldr: this extracts the ground truth in the same format as the model output
            batch_size = len(pred_output)
            # get gt_output and timestamps to be in the same format as pred_output
            timestamps = [{} for _ in range(batch_size)]
            subtask_index = [{} for _ in range(batch_size)]
            gt_output = [{} for _ in range(batch_size)]

            # collect ground truth
            for taskname, spec in pl_module.model.readout.task_specs.items():
                taskid = Output.from_string(taskname).value

                # get the mask of tokens that belong to this task
                mask = batch["output_task_index"] == taskid

                if not torch.any(mask):
                    # there is not a single token for this task, so we skip
                    continue

                # we need to distribute the outputs to their respective samples
                token_batch = torch.where(mask)[0]
                batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
                for i in range(len(batch_i)):
                    timestamps[batch_i[i]][taskname] = (
                        batch["output_timestamps"][mask][token_batch == i]
                        + absolute_starts[i]
                    )
                    subtask_index[batch_i[i]][taskname] = output_subtask_index[
                        taskname
                    ][(token_batch == i).detach().cpu()]
                    gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                        token_batch == i
                    ]

            # register all of the data
            for i in range(batch_size):
                session_id = session_ids[i]

                if session_id not in session_pred_output:
                    session_pred_output[session_id] = {}
                    session_gt_output[session_id] = {}
                    session_timestamp[session_id] = {}
                    session_subtask_index[session_id] = {}

                for taskname, pred_values in pred_output[i].items():
                    if taskname not in session_pred_output[session_id]:
                        session_pred_output[session_id][
                            taskname
                        ] = pred_values.detach().cpu()
                        session_gt_output[session_id][taskname] = (
                            gt_output[i][taskname].detach().cpu()
                        )
                        session_timestamp[session_id][taskname] = (
                            timestamps[i][taskname].detach().cpu()
                        )
                        session_subtask_index[session_id][taskname] = (
                            subtask_index[i][taskname].detach().cpu()
                        )
                    else:
                        session_pred_output[session_id][taskname] = torch.cat(
                            (session_pred_output[session_id][taskname],
                            pred_values.detach().cpu(),)
                        )
                        session_gt_output[session_id][taskname] = torch.cat(
                            (session_gt_output[session_id][taskname],
                            gt_output[i][taskname].detach().cpu(),)
                        )
                        session_timestamp[session_id][taskname] = torch.cat(
                            (session_timestamp[session_id][taskname],
                            timestamps[i][taskname].detach().cpu(),)
                        )
                        session_subtask_index[session_id][taskname] = torch.cat(
                            (session_subtask_index[session_id][taskname],
                            subtask_index[i][taskname].detach().cpu(),)
                        )

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
            raise NotImplementedError("Gathering is not yet fully tested.")
            session_timestamp = gather_concat_dict(session_timestamp)
            session_gt_output = gather_concat_dict(session_gt_output)
            session_pred_output = gather_concat_dict(session_pred_output)
            session_subtask_index = gather_concat_dict(session_subtask_index)

        if trainer.local_rank != 0:
            return

        metrics = dict()
        for session_id in tqdm(
            session_gt_output,
            desc=f"Compiling metrics @ Epoch {trainer.current_epoch}",
            disable=(trainer.local_rank != 0),
        ):
            for taskname in session_gt_output[session_id]:
                gt = session_gt_output[session_id][taskname]
                pred = session_pred_output[session_id][taskname]
                timestamps = session_timestamp[session_id][taskname]
                subtask_index = session_subtask_index[session_id][taskname]

                # pool
                output_type = pl_module.model.readout.task_specs[taskname].type
                if output_type == OutputType.CONTINUOUS:
                    pred = avg_pool(timestamps, pred)
                    gt = avg_pool(timestamps, gt)
                elif output_type in [
                    OutputType.BINARY,
                    OutputType.MULTINOMIAL,
                    OutputType.MULTILABEL,
                ]:
                    assert gt.ndim == 2
                    assert gt.shape[1] == 1, "Only one label per trial is supported."
                    if gt.shape[0] > 1:
                        assert all(
                            torch.all(gt[0] == x) for x in gt
                        ), "All labels must be the same for a trial."
                    gt = gt[0].unsqueeze(0)
                    pred = pred.mean(dim=0).unsqueeze(0)

                # Compute metrics
                task_spec = pl_module.model.readout.task_specs[taskname]

                # Resolve the appropriate loss function.
                metrics[f"val/{session_id}/{str(taskname.lower())}/r2"] = (
                    compute_loss_or_metric("r2", task_spec.type, pred, gt, 1.0).item()
                )

        pl_module.log_dict(metrics)
        logging.info(f"Logged {len(metrics)} validation metrics.")

        metrics = pd.DataFrame(metrics, index=["metric"]).T
        if pl_module.tb is not None:
            pl_module.tb.add_text("val_metrics", metrics.to_markdown())
        if pl_module.wandb is not None:
            pl_module.wandb.log({"val_metrics": wandb.Table(dataframe=metrics)})

        return metrics


def avg_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    unique_timestamps, indices = torch.unique(timestamps, return_inverse=True)
    averages = torch.zeros((len(unique_timestamps), values.shape[1]))

    for i in range(len(unique_timestamps)):
        group_values = values[indices == i]
        averages[i] = torch.mean(group_values, dim=0)
    return averages
