import collections
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchtext
import yaml
from einops import repeat
from torchtyping import TensorType

from kirby.data import Data
from kirby.data.data import RegularTimeSeries
from kirby.taxonomy import StringIntEnum
from kirby.taxonomy.taxonomy import DecoderSpec, RecordingTech
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
        self.chunk_info, self.session_names, self.unit_names = self.look_for_files()

        self.sequence_len_file = sequence_len_file

    def look_for_files(self) -> Tuple[List[Dict], List[str], List]:
        chunk_info = []
        session_names = []
        unit_names = []

        for i, included_datasets in enumerate(self.include):
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
            # subject and session, but we could make selection more flexible in the 
            # future.
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

            assert len(sessions) > 0, f"No sessions found for {i}'th dataset included"

            session_names += [session["id"] for session in sessions]

            # Now we get the chunk-level information.
            for session in sessions:
                for trial in session["trials"]:
                    for chunk in trial["chunks"][self.split]:
                        iomap = {k: session[k] for k in ["inputs", "outputs", "stimuli", "task"]}
                        chunk_info.append(
                            {
                                "filename": (
                                    Path(self.root)
                                    / selection["dandiset"]
                                    / self.split
                                    / f"{chunk['id']}.pt"
                                ),
                                "iomap": iomap,
                                "description": included_datasets,
                            }
                        )

        all_filenames = [info["filename"] for info in chunk_info]

        assert len(set(all_filenames)) == len(
            all_filenames
        ), f"Overlapping selection criteria for {self.split} datasets"

        unit_names = list(set(unit_names))
        return chunk_info, session_names, unit_names

    def __getitem__(self, item):
        info = self.chunk_info[item]
        data = torch.load(info["filename"])
        # apply transform
        if self.transform is not None:
            data = self.transform(data)
        data.description = info["description"]
        data.iomap = info["iomap"]
        return data

    def __len__(self):
        return len(self.chunk_info)

    def few_shot(self, num_samples, shuffle=True):
        assert num_samples <= len(
            self
        ), f"Cannot sample {num_samples} from dataset of length {len(self)}"
        if shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = torch.arange(len(self))
        self.chunk_info = [self.chunk_info[i] for i in indices[:num_samples]]
        self.session_names = [self.session_names[i] for i in indices[:num_samples]]
        return self

    def augment_for_batchsize(self, batch_size: int):
        curr_len = len(self)
        if curr_len < batch_size:
            self.chunk_info = self.chunk_info * (1 + ((batch_size - 1) // curr_len))
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

@dataclass
class PaddedGrouping:
    spike_timestamps: TensorType["batch", "nspikes"]
    # Spike ids are resolved with respect to unit names in the collator.
    spike_ids: TensorType["batch", "nspikes"]
    spike_type: TensorType["batch", "nspikes"]

    # True for all real spikes and start/end events
    input_mask: TensorType["batch", "nspikes"]
    # TODO: remove mask, it's redundant with input_mask

    latent_timestamps: TensorType["batch", "nlatents"]
    latent_ids: TensorType["batch", "nlatents"]

    # Pytorch geometric style
    output_task_index: Dict[str, TensorType["*Bout"]]
    output_timestamps: Dict[str, TensorType["*Bout", "*ntout"]]
    output_values: Dict[str, TensorType["*Bout", "*ntout", "*nchannelsout"]]
    # We absorb the mask into the weight
    output_weight: Dict[str, TensorType["*Bout", "*ntout"]]

    session_names: List[str]

    # We use only the central channel for now.
    spike_waveforms: Optional[TensorType["batch", "nspikes", "nt"]] = None

    # We represent average waveforms.
    average_waveforms: Optional[TensorType["nunits", "nt"]] = None

    # LFPs, when available, are referenced with respect to spikes via nearest neighbour
    # interpolation. Multiple bands are referenced in parallel.
    lfps: Optional[TensorType["batch", "nspikes", "lfp_bands"]] = None

    # Q: should we have masks for spike waveforms and for lfps?
    
    def to_dict(self):
        return dataclasses.asdict(self)


def check_include_exclude(description, key):
    if "include_input" in description:
        return key in description["include_input"]
    elif "exclude_input" in description.keys():
        return key not in description["exclude_input"]
    else:
        return True
    
def resolve(data, key) -> torch.Tensor:
    # Split key by dots, resolve using getattr
    components = key.split('.')
    for c in components:
        try:
            data = getattr(data, c)
        except AttributeError:
            raise AttributeError(f"Could not resolve {key} in data (specifically, at level {c}))")
    return data



class Collate:
    def __init__(
        self,
        num_latents_per_step: int = 1,
        step=1.0,
        behavior_type_weight=None,
        reweight: bool = False,
        sequence_length=1.0,
        unit_vocab: Optional[torchtext.vocab.Vocab]=None,
        decoder_registry: Optional[Dict[str, DecoderSpec]]=None,
    ):
        """Stack datasets into a batch.

        Args:
            num_latents_per_step: Number of latents per step.
            step: Step size between latents in seconds.
            behavior_type_weight: Weight for each behavior type.
            reweight: Whether to reweight the loss so that each sequence has the same weight.
            sequence_length: Length of each sequence in seconds.

        Note that there are necessarily sequence_length / step * num_latents_per_step latent tokens.
        """
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.behavior_type_weight = behavior_type_weight
        self.reweight = reweight
        if unit_vocab is None:
            raise NotImplementedError("Unit vocab is required")
        self.unit_vocab = unit_vocab
        # Make sure that the unit vocab has a mapping for NA and that it corresponds to 0
        assert self.unit_vocab.forward(["NA"])[0] == 0

        # TODO: remove sequence_length from the parameters and read the sequence length 
        # from the data instead.
        self.sequence_length = sequence_length
        if decoder_registry is None:
            raise NotImplementedError("Decoder registry is required")
        self.decoder_registry = decoder_registry


    def __call__(self, batch: List[Data]) -> Dict[str, Union[torch.Tensor, List]]:
        # Deal with the inputs first
        num_tokens = [
            len(data.spikes) + len(data.units.unit_name) * 2 for data in batch
        ]
        max_num_tokens = next_multiple_of_8(max(num_tokens))
        max_num_units = len(self.unit_vocab)

        spike_timestamps = torch.zeros(
            (len(batch), max_num_tokens), dtype=torch.float32
        )
        spike_type = torch.zeros((len(batch), max_num_tokens), dtype=torch.long)
        spike_ids = torch.zeros((len(batch), max_num_tokens), dtype=torch.long)
        mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)

        # Q: do we have waveforms, and do we have LFPs?
        has_spike_waveforms = False
        has_average_waveforms = False
        has_lfps = False

        max_spike_waveform_size = 0
        max_average_waveform_size = 0
        max_nbands = 0

        max_output_tokens = collections.defaultdict(int)
        output_dims = collections.defaultdict(int)
        output_task_index = collections.defaultdict(list)

        for i, data in enumerate(batch):
            # Two conditions: the data exists, and it's been requested.
            data.has_spike_waveforms = False
            data.has_average_waveforms = False
            data.has_lfps = False
            if (str(RecordingTech.UTAH_ARRAY_WAVEFORMS)) in data.iomap['inputs'].keys():
                check = check_include_exclude(data.description, str(RecordingTech.UTAH_ARRAY_WAVEFORMS))
                if check:
                    max_spike_waveform_size = max(max_spike_waveform_size, data.spikes.waveforms.shape[1])
                    data.has_spike_waveforms = check
                    has_spike_waveforms = check

            if (str(RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS)) in data.iomap['inputs'].keys():
                check = check_include_exclude(data.description, str(RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS))
                if check:
                    max_average_waveform_size = max(max_average_waveform_size, data.spikes.waveforms.shape[1])
                    data.has_average_waveforms = check
                    has_average_waveforms = check
                
            if (str(RecordingTech.UTAH_ARRAY_LFPS)) in data.iomap['inputs'].keys():
                check = check_include_exclude(data.description, str(RecordingTech.UTAH_ARRAY_LFPS))
                if check:
                    max_nbands = max(max_nbands, len(data.lfp_metadata.bands))
                    data.has_lfps = check
                    has_lfps = check

            # Now we deal with the outputs.
            for metric in data.description['metrics']:
                # This is just a sketch of how we might do this-needs some work.
                key = metric['output_key']
                output_dims[key] = self.decoder_registry[key].dim
                value = resolve(data, self.decoder_registry[key].value_key)
                max_output_tokens[key] = max(max_output_tokens[key], value.shape[0])
                output_task_index[key].append(i)

        if has_spike_waveforms:
            spike_waveforms = torch.zeros(len(batch), max_num_tokens, max_spike_waveform_size)

        if has_average_waveforms:
            average_waveforms = torch.zeros(max_num_units, max_average_waveform_size)

        if has_lfps:
            lfps = torch.zeros((len(batch), max_num_tokens, max_nbands), dtype=torch.float32)

        
        output_timestamps = {}
        output_values = {}
        output_weight = {}
        output_task_index = dict(output_task_index)
        output_offset = {}

        for key in output_dims.keys():
            output_timestamps[key] = torch.zeros(len(output_task_index[key]), max_output_tokens[key])
            output_values[key] = torch.zeros(len(output_task_index[key]), max_output_tokens[key], output_dims[key])
            output_weight[key] = torch.zeros(len(output_task_index[key]), max_output_tokens[key])
            output_offset[key] = 0

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
        # output_mask = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.bool)

        # fill values
        for i, data in enumerate(batch):
            # add spike events
            spikes = data.spikes

            # Annoyingly, we have to keep spike names as a list of strings, as PyTorch
            # does not support string tensors. We will convert to a tensor later.
            mapped_spikes = self.unit_vocab.forward(list(spikes.names) + list(data.units.unit_name) + list(data.units.unit_name))
            spike_ids[i, : len(mapped_spikes)] = torch.Tensor(mapped_spikes)
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
            input_mask[i, : len(spikes) + len(units) * 2] = True

            # Add waveforms
            if data.has_spike_waveforms:
                spike_waveforms[i, : len(spikes), : spikes.waveforms.shape[1]] = spikes.waveforms

            # Add average waveforms
            if data.has_average_waveforms:
                inverse_slot = self.unit_vocab(units.tolist())
                average_waveforms[inverse_slot, :] = torch.Tensor(data.units.average_waveform)

            # Add local field potentials
            if data.has_lfps:
                # We do a local, nearest neighbour lookup to find the LFPs corresponding to each spike.
                # We use the LFP index to do so.
                # Build a map from unit name to LFP index.
                lfp_channel_to_idx = {x: i for i, x in enumerate(data.lfp_metadata.channels)}
                unit_to_channel = {u: lfp_channel_to_idx[c] for u, c in zip(data.units.unit_name, data.units.channel_name)}
                lfp_idx = [unit_to_channel[u] for u in spikes.names]

                # Now find the corresponding offset using nearest neighbour.
                # Note that this is only safe to do on a RegularTimeSeries
                assert isinstance(data.lfps, RegularTimeSeries)
                lfp_approx_index = data.lfps.sampling_rate * (spikes.timestamps - data.lfps.timestamps[0])
                lfp_tidx = torch.clip(torch.round(lfp_approx_index), 0, data.lfps.lfp.shape[1]).to(torch.long)
                assert len(lfp_idx) == len(lfp_tidx)
                selected_lfp = data.lfps.lfp[lfp_tidx, lfp_idx, :]

                lfps[i, : selected_lfp.shape[0], : selected_lfp.shape[1]] = selected_lfp

            # Now we deal with the outputs.
            for metric in data.description['metrics']:
                # This is just a sketch of how we might do this-needs some work.
                key = metric['output_key']
                timestamps = resolve(data, self.decoder_registry[key].timestamp_key)
                value = resolve(data, self.decoder_registry[key].value_key)

                output_timestamps[key][output_offset[key], : timestamps.shape[0]] = timestamps
                output_values[key][output_offset[key], : timestamps.shape[0], :] = value
                # Right now, the weight is a simple mask, but we could recover the 
                # previous functionality if necessary.
                output_weight[key][output_offset[key], : timestamps.shape[0]] = 1.0
                output_offset[key] += 1

        session_names = [data.session for data in batch]

        extras = {}
        if has_spike_waveforms:
            extras["spike_waveforms"] = spike_waveforms

        if has_average_waveforms:
            extras["average_waveforms"] = average_waveforms

        if has_lfps:
            extras["lfps"] = lfps

        data = PaddedGrouping(
            spike_timestamps=spike_timestamps,
            spike_ids=spike_ids,
            spike_type=spike_type,
            input_mask=input_mask,
            latent_timestamps=latent_timestamps,
            latent_ids=latent_ids,
            output_task_index=output_task_index,
            output_timestamps=output_timestamps,
            output_values=output_values,
            output_weight=output_weight,
            session_names=session_names,
            **extras
        )
        return data
