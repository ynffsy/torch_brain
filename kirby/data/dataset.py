import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import yaml
from einops import repeat

from kirby.data import Data
from kirby.taxonomy import StringIntEnum
from kirby.utils import logging

log = logging(header="DATASET", header_color="red")


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, root, split, include=None, transform=None, sequence_len_file=None
    ):
        super().__init__()
        self.root = root

        assert split in ["train", "valid", "test", "finetune"]
        self.split = split

        if include is None:
            raise ValueError("Please specify the datasets to include")

        self.include = include
        self.transform = transform
        self.filenames, self.session_names, self.unit_names = self.look_for_files()

        self.sequence_len_file = sequence_len_file

    def look_for_files(self) -> Tuple[List[Path], List[str], List[str]]:
        session_names = []
        unit_names = []
        filenames = []

        for included_datasets in self.include:
            selection = included_datasets["selection"]
            if selection.get("dandiset", "") == "":
                raise ValueError(
                    f"Please specify a dandiset to include under {self.split}_datasets"
                )

            description_file = os.path.join(
                self.root, selection["dandiset"], "description.yaml"
            )

            try:
                with open(description_file, "r") as f:
                    description = yaml.load(f, Loader=yaml.CLoader)

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find description file {description_file}"
                )

            # Get a list of all the potentially chunks in this dataset.
            sortsets = description["sortsets"]

            # Perform selection. Right now, we are limiting ourselves to sortset,
            # subject and session, but we could make selection more flexible in the future.
            sel_sortset = selection.get("sortset", None)
            sel_subject = selection.get("subject", None)
            sel_session = selection.get("session", None)

            # First, we get the sortset-level information.
            if sel_sortset is not None:
                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] == sel_sortset
                ]

            if sel_subject is not None:
                sortsets = [
                    sortset for sortset in sortsets if sortset["subject"] == sel_subject
                ]

            # Note that this logic may result in adding two many slots but that's fine.
            unit_names += [x for sortset in sortsets for x in sortset["units"]]

            # Now we get the session-level information.
            sessions = sum([sortset["sessions"] for sortset in sortsets], [])
            if sel_session is not None:
                sessions = [
                    session for session in sessions if session["id"] == sel_session
                ]

            session_names += [session["id"] for session in sessions]

            # Now we get the chunk-level information.
            for session in sessions:
                for trial in session["trials"]:
                    for chunk in trial["chunks"][self.split]:
                        filenames.append(
                            Path(self.root)
                            / selection['dandiset']
                            / self.split
                            / f"{chunk['id']}.pt"
                        )

        unit_names = list(set(unit_names))
        return filenames, session_names, unit_names

    def __getitem__(self, item):
        data = torch.load(self.filenames[item])
        # apply transform
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.filenames)

    def few_shot(self, num_samples, shuffle=True):
        assert num_samples <= len(
            self
        ), f"Cannot sample {num_samples} from dataset of length {len(self)}"
        if shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = torch.arange(len(self))
        self.filenames = [self.filenames[i] for i in indices[:num_samples]]
        self.session_names = [self.session_names[i] for i in indices[:num_samples]]
        return self

    def augment_for_batchsize(self, batch_size: int):
        curr_len = len(self)
        if curr_len < batch_size:
            self.filenames = self.filenames * (1 + ((batch_size - 1) // curr_len))
            self.session_names = self.session_names * (
                1 + ((batch_size - 1) // curr_len)
            )
        return self

    def get_sequence_len(self):
        if self.sequence_len_file is None:
            # warn that compute can be slow
            # also if transform is used, this will be wrong
            log.warn(
                "Computing sequence lengths can be slow, consider specifying a sequence length file"
            )
            sequence_len = np.array([len(data.spikes) for data in self])
        else:
            # load npy file
            sequence_len = np.load(self.sequence_len_file)
        return sequence_len


def next_multiple_of_8(x):
    remainder = x % 8
    if remainder == 0:
        return x
    else:
        return x + (8 - remainder)


class SpikeType(StringIntEnum):
    UNIT = 0
    START = 1
    END = 2


class Collate:
    def __init__(
        self,
        num_latents_per_step=1,
        step=1.0,
        behavior_type_weight=None,
        reweight=False,
        sequence_length=1.0,
    ):
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.behavior_type_weight = behavior_type_weight
        self.reweight = reweight
        self.sequence_length = sequence_length

    def __call__(self, batch: List[Data]) -> Dict[str, Union[torch.Tensor, List]]:
        # make spike tensors
        num_tokens = [len(data.spikes) + len(data.units.unit_name) * 2 for data in batch]
        max_num_tokens = next_multiple_of_8(max(num_tokens))

        # print(isinstance(batch[0].spikes.timestamps, torch.Tensor))
        # print(batch[0].spikes.timestamps.dtype)
        # print(batch[0].spikes.timestamps.device)
        # print(isinstance(batch[0].spikes.timestamps, np.ndarray))
        # print("---")

        spike_timestamps = torch.zeros(
            (len(batch), max_num_tokens), dtype=torch.float32
        )
        spike_names = []
        spike_type = torch.zeros((len(batch), max_num_tokens), dtype=torch.long)
        mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)

        num_output_tokens = [len(data.behavior.timestamps) for data in batch]
        max_num_output_tokens = next_multiple_of_8(max(num_output_tokens))

        # make behavior tensors
        output_timestamps = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.float32
        )
        output_values = torch.empty(
            (len(batch), max_num_output_tokens, 2), dtype=torch.float32
        ).fill_(1e6)
        output_weight = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.float32
        )
        output_stage = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.long
        )

        # make latent tensors
        latent_timestamps = (
            torch.arange(0, self.sequence_length, self.step) + self.step / 2
        )
        latent_ids = torch.arange(self.num_latents_per_step, dtype=torch.long)
        num_timestamps = len(latent_timestamps)
        latent_timestamps = repeat(
            latent_timestamps, "t -> b (t u)", b=len(batch), u=len(latent_ids)
        )
        latent_ids = repeat(latent_ids, "u -> b (t u)", b=len(batch), t=num_timestamps)

        num_timestamps = latent_timestamps.size(1)

        # make attn masks
        input_mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)
        output_mask = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.bool)

        # fill values
        for i, data in enumerate(batch):
            # add spike events
            spikes = data.spikes

            # Annoyingly, we have to keep spike names as a list of strings, as PyTorch
            # does not support string tensors. We will convert to a tensor later.
            spike_names.append(
                list(spikes.names)
                + list(data.units.unit_name)
                + list(data.units.unit_name)
                + ["NA"] * (max_num_tokens - len(spikes) - len(data.units.unit_name) * 2)
            )
            spike_timestamps[i, : len(spikes)] = spikes.timestamps
            mask[i, : len(spikes)] = True
            # add artificial start and end of trial events to each unit
            units = data.units.unit_name
            start, end = data.start, data.end
            # assume that aligned with start and end
            start, end = 0.0, end - start
            spike_timestamps[i, len(spikes) : len(spikes) + len(units)] = start
            spike_timestamps[
                i, len(spikes) + len(units) : len(spikes) + len(units) * 2
            ] = end
            spike_type[i, len(spikes) : len(spikes) + len(units)] = int(SpikeType.START)
            spike_type[
                i, len(spikes) + len(units) : len(spikes) + len(units) * 2
            ] = int(SpikeType.END)

            # make output
            output = data.behavior
            output_timestamps[i, : len(output.timestamps)] = output.timestamps
            output_values[i, : len(output.timestamps)] = output.hand_vel
            output_mask[i, : len(output.timestamps)] = True

            behavior_type = (
                output.type if hasattr(output, "type") else output.behavior_type
            )
            output_stage[i, : len(output.timestamps)] = behavior_type
            output_weight[i, : len(output.timestamps)] = (
                self.behavior_type_weight[behavior_type]
                if self.behavior_type_weight is not None
                else 1.0
            )
            # reweight so that each trial is equally important
            if self.reweight:
                output_weight[i] *= max_num_output_tokens / len(output.timestamps)

            # update masks
            input_mask[i, : len(spikes) + len(units) * 2] = True

        # session id
        session_names = [data.session for data in batch]

        data = dict(
            spike_timestamps=spike_timestamps,
            spike_names=spike_names,
            spike_type=spike_type,
            # TODO: clean this up. Why is there a mask distinct from input_mask?
            mask=mask,
            input_mask=input_mask,
            output_timestamps=output_timestamps,
            output_values=output_values,
            output_weight=output_weight,
            output_mask=output_mask,
            output_stage=output_stage,
            # repeat and pin_memory don't play well, hence we clone
            latent_timestamps=torch.clone(latent_timestamps),
            latent_id=torch.clone(latent_ids),
            session_names=session_names,
        )
        return data