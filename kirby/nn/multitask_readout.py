from typing import Dict, List, Optional, Tuple, Union
import copy

from pydantic.dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from kirby.taxonomy import DecoderSpec, Decoder
from kirby.data.collate import collate, chain, track_batch
from kirby.nn import compute_loss_or_metric


class MultitaskReadout(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        task_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        # Create a bunch of projection layers. One for each task
        self.projections = nn.ModuleDict({})
        for taskname, spec in task_specs.items():
            self.projections[taskname] = nn.Linear(latent_dim, spec.dim)

        # Need task specs layer to decide loss type
        self.task_specs = task_specs

    def forward(
        self,
        output_latents: Union[TensorType["batch", "max_ntout", "dim"], TensorType["total_ntout", "dim"]],
        output_task_index: Union[TensorType["batch", "max_ntout"], TensorType["total_ntout"]],
        output_batch_index: Optional[TensorType["max_ntout"]] = None,
        output_values: Dict[str, TensorType["*ntout_task", "*nchannelsout"]] = None,
        output_weights: Dict[str, TensorType["*ntout_task"]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        Union[None, torch.Tensor],
        Union[None, Dict[str, torch.Tensor]],
    ]:
        """
        Args:
            output_latents: Outputs of the last transformer layer.
            output_task_indices: Task index for each token in (batch, max_ntout).
            output_values: Ground-truth values for loss computation.
                output_values[task] is the ground truth value for the task
            output_weights: Sample-wise weights for loss computation.
                output_weights[task] is the weight for a given task.
        """

        if output_batch_index is not None:
            # Inputs were chained, make sure input dimensions make sense
            assert output_latents.dim() == 2
            assert output_task_index.dim() == 1
            assert output_batch_index.dim() == 1
            batch_size = output_batch_index.max().item() + 1
        else:
            # Inputs were not chained, make sure input dimensions make sense
            assert output_latents.dim() == 3
            assert output_task_index.dim() == 2
            batch_size = output_latents.shape[0]

        outputs = [{} for _ in range(batch_size)]
        taskwise_loss = {}
        loss = torch.tensor(0, device=output_latents.device, dtype=torch.float32)

        for taskname, spec in self.task_specs.items():
            # the taskid is a universal unique identifier for the task
            taskid = Decoder.from_string(taskname).value

            # get the mask of tokens that belong to this task
            mask = output_task_index == taskid
            
            if not torch.any(mask):
                # there is not a single token for this task, so we skip
                continue
            
            # apply the projection
            task_output = self.projections[taskname](output_latents[mask])

            if output_values is not None:
                target = output_values[taskname]
                
                weights = 1.0
                if taskname in output_weights and output_weights[taskname] is not None:
                    weights = output_weights[taskname]

                taskwise_loss[taskname] = compute_loss_or_metric(
                    spec.loss_fn, spec.type, task_output, target, weights
                )

            # we need to distribute the outputs to their respective samples
            if output_batch_index is None:
                token_batch = torch.where(mask)[0]
            else:
                # Inputs where chained, and we have batch-indices for each token
                token_batch = output_batch_index[mask]

            batch, token_batch = torch.unique(token_batch, return_inverse=True)
            for i in range(len(batch)):
                outputs[batch[i]][taskname] = task_output[token_batch == i]

            if output_values is not None:
                # Since we calculate a mean across all elements, scale by the number of
                # items in the batch so we don't get wild swings in loss depending on
                # whether we have large or small numbers of non-dominant classes.
                loss = loss + taskwise_loss[taskname] * len(batch)

        loss = loss / batch_size

        if output_values is None:
            return outputs, None, None

        return outputs, loss, taskwise_loss


def parse_multitask_readout_config(config):
    if "multitask_readout" not in config:
        return None
    
    assert len(config["multitask_readout"]) > 1, "At least one decoder must be defined."

    decoder_config_list = []
    for decoder_config in  config["multitask_readout"]:
        if "decoder_id" not in decoder_config:
            raise ValueError("a decoder_id must be defined.")
        
        decoder_id = decoder_config["decoder_id"]

        decoder_registry[decoder_id]

        # allow timestamp_key, value_key, task_key, subtask_key to be overwritten
        for key in ["timestamp_key", "value_key", "task_key", "subtask_key"]:
            if key not in decoder_config:
                decoder_config[key] = decoder_config[key]

        # get weights
        task_weights = decoder_config.get("task_weights", {})
        subtask_weights = decoder_config.get("subtask_weights", {})
    pass


def prepare_for_multitask_readout(
    data, decoder_registry: Dict[str, DecoderSpec],
):
    decoder_index = list()
    timestamps = list()
    values = dict()
    task_index = dict()
    subtask_index = dict()
    weights = dict()
    
    config = data.config["multitask_readout"]

    # for metric in data.description["metrics"]:
    for decoder in config:
        key = decoder["decoder_id"]

        decoder_index.append(Decoder.from_string(key).value)
        timestamps.append(data.get_nested_attribute(decoder["timestamp_key"]))
        values[key] = data.get_nested_attribute(decoder["value_key"])
        # here we assume that we won't be running a model at float64 precision
        if values[key].dtype == np.float64:
            values[key] = values[key].astype(np.float32)

        if decoder["task_index"] is not None:
            task_index[key] = data.get_nested_attribute(decoder["task_index"])
        else:
            task_index[key] = np.zeros(len(values[key]), dtype=np.int64)
        
        if decoder["subtask_key"] is not None:
            subtask_index[key] = data.get_nested_attribute(decoder["subtask_key"])
        else:
            subtask_index[key] = np.zeros(len(values[key]), dtype=np.int64)
        
        weights = ...

        weights_ = torch.tensor(
            [weight_registry.get(int(x.item()), -1.0) for x in subtask_index[key]]
        )

        # Either we have weights for all or for none (implicitly, everything
        # has a weight of 1 in that case). There shouldn't be any
        # in-between cases, which would mean there's an undefined behaviour.
        if torch.any(weights_ == -1.0) and not torch.all(weights_ == -1.0):
            idx = torch.where(weights_ == 0)[0][0]
            raise ValueError(
                f"Could not find weights for behavior #{subtask_index[key][idx]}"
            )

        weights_[weights_ == -1.0] = 1.0
        weights[key] = weights_ * metric.get("weight", 1.0)

    # chain
    timestamps, batch = collate(
        [
            (chain(timestamps[i]), track_batch(timestamps[i]))
            for i in range(len(timestamps))
        ]
    )
    decoder_index = torch.tensor(decoder_index)[batch]

    return timestamps, decoder_index, values, weights, subtask_index
