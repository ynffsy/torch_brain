from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from kirby.taxonomy import DecoderSpec, Output
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
        output_latents: TensorType["batch", "max_ntout", "latent_dim"],
        output_task_index: TensorType["batch", "max_ntout"],
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

        outputs = [{} for _ in range(output_latents.shape[0])]
        taskwise_loss = {}
        loss = torch.tensor(0, device=output_latents.device, dtype=torch.float32)

        for taskname, spec in self.task_specs.items():
            # the taskid is a universal unique identifier for the task
            taskid = Output.from_string(taskname).value

            # get the mask of tokens that belong to this task
            mask = output_task_index == taskid
            
            if not torch.any(mask):
                # there is not a single token for this task, so we skip
                continue
            
            # apply the projection
            task_output = self.projections[taskname](output_latents[mask])

            if output_values is not None:
                target = output_values[taskname]
                
                weights = None
                if taskname in output_weights and output_weights[taskname] is not None:
                    weights = output_weights[taskname]
                weights = weights if weights is not None else 1.0

                taskwise_loss[taskname] = compute_loss_or_metric(
                    spec.loss_fn, spec.type, task_output, target, weights
                )

            # we need to distribute the outputs to their respective samples
            token_batch = torch.where(mask)[0]
            batch, token_batch = torch.unique(token_batch, return_inverse=True)
            for i in range(len(batch)):
                outputs[batch[i]][taskname] = task_output[token_batch == i]

            if output_values is not None:
                # Since we calculate a mean across all elements, scale by the number of
                # items in the batch so we don't get wild swings in loss depending on
                # whether we have large or small numbers of non-dominant classes.
                loss = loss + taskwise_loss[taskname] * len(batch)

        loss = loss / output_latents.shape[0]

        if output_values is None:
            return outputs, None, None

        return outputs, loss, taskwise_loss

      
def prepare_for_multitask_readout(
    data, decoder_registry: Dict[str, DecoderSpec], weight_registry
):
    timestamps = list()
    task_index = list()
    values = dict()
    weights = dict()

    for metric in data.description["metrics"]:
        key = metric["output_key"]

        task_index.append(Output.from_string(key).value)
        timestamps.append(data.get_nested_attribute(decoder_registry[key].timestamp_key))

        values[key] = data.get_nested_attribute(decoder_registry[key].value_key)

        try:
            behavior_type = data.get_nested_attribute(decoder_registry[key].behavior_type_key)
        except AttributeError:
            behavior_type = np.zeros(len(values[key]), dtype=np.int64)

        weights_ = torch.tensor(
            [weight_registry.get(int(x.item()), -1.0) for x in behavior_type]
        )

        # Either we have weights for all or for none (implicitly, everything
        # has a weight of 1 in that case). There shouldn't be any
        # in-between cases, which would mean there's an undefined behaviour.
        if torch.any(weights_ == -1.0) and not torch.all(weights_ == -1.0):
            idx = torch.where(weights_ == 0)[0][0]
            raise ValueError(
                f"Could not find weights for behavior #{behavior_type[idx]}"
            )

        weights_[weights_ == -1.0] = 1.0
        weights[key] = torch.tensor(weights_) * metric.get("weight", 1.0)

    # chain
    timestamps, batch = collate(
        [(chain(timestamps[i]), track_batch(timestamps[i])) for i in range(len(timestamps))]
    )
    task_index = torch.tensor(task_index)[batch]

    return timestamps, task_index, values, weights
