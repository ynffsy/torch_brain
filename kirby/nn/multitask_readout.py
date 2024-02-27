from typing import Dict, List, Optional, Tuple, Union
import copy

from pydantic.dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from kirby.taxonomy import DecoderSpec, Decoder, Task
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


def prepare_for_multitask_readout(
    data, decoder_registry: Dict[str, DecoderSpec],
):
    decoder_index = list()
    timestamps = list()
    values = dict()
    # task_index = dict()
    subtask_index = dict()
    weights = dict()
    
    config = data.config["multitask_readout"]

    for decoder in config:
        key = decoder["decoder_id"]
        weight = decoder.get("weight", 1.0)
        subtask_weights = decoder.get("subtask_weights", {})

        decoder = decoder_registry[key].__dict__ | decoder  # config overrides registry

        decoder_index.append(Decoder.from_string(key).value)
        timestamps.append(data.get_nested_attribute(decoder["timestamp_key"]))
        
        values[key] = data.get_nested_attribute(decoder["value_key"])
        # here we assume that we won't be running a model at float64 precision
        # TODO do this in decoder spec? 
        if values[key].dtype == np.float64:
            values[key] = values[key].astype(np.float32)

        # if decoder["task_index"] is not None:
        #     task_index[key] = data.get_nested_attribute(decoder["task_index"])
        # else:
        #     task_index[key] = np.zeros(len(values[key]), dtype=np.int64)
        
        if decoder["subtask_key"] is not None:
            subtask_index[key] = data.get_nested_attribute(decoder["subtask_key"])
            num_subtasks = Task.from_string(list(subtask_weights.keys())[0]).max_value()
            subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
            for subtask, subtask_weight in subtask_weights.items():
                subtask_weight_map[Task.from_string(subtask).value] = subtask_weight
            
            subtask_weight_map *= weight
            weights[key] = subtask_weight_map[subtask_index[key]]
        else:
            subtask_index[key] = np.zeros(len(values[key]), dtype=np.int64)
            weights[key] = np.ones(len(values[key]), dtype=np.float32) * weight

    # chain
    timestamps, batch = collate(
        [
            (chain(timestamps[i]), track_batch(timestamps[i]))
            for i in range(len(timestamps))
        ]
    )
    decoder_index = torch.tensor(decoder_index)[batch]

    return timestamps, decoder_index, values, weights, subtask_index
