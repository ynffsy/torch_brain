import collections
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import warnings
import msgpack
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchtext
from einops import repeat
from torchtyping import TensorType

from kirby.data import Data
from kirby.data.data import RegularTimeSeries
from kirby.taxonomy import StringIntEnum, description_helper
from kirby.taxonomy.taxonomy import DecoderSpec, RecordingTech


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        include: List[Dict[str, Any]] = None,
        transform=None,
    ):
        super().__init__()
        self.root = root

        assert split in ["train", "valid", "test", "finetune"]
        self.split = split

        if include is None:
            raise ValueError("Please specify the datasets to include")

        self.include = include
        self.transform = transform
        (
            self.chunk_info,
            self.session_names,
            self.unit_names,
        ) = self.look_for_files()

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
                self.root, selection["dandiset"], "description.mpk"
            )

            try:
                with open(description_file, "rb") as f:
                    description = msgpack.load(
                        f, object_hook=description_helper.decode_datetime
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find description file {description_file}. This error "
                    "might be due to running an old pipeline that generates a "
                    "description.yaml file. Try running the appropriate snakemake "
                    "pipeline to generate the msgpack (mpk) file instead."
                )

            # Get a list of all the potentially chunks in this dataset.
            sortsets = description["sortsets"]

            # Perform selection. Right now, we are limiting ourselves to sortset,
            # subject and session, but we could make selection more flexible in the
            # future.
            sel_sortset = selection.get("sortset", None)
            sel_sortset_lte = selection.get("sortset_lte", None)
            sel_subject = selection.get("subject", None)
            sel_session = selection.get("session", None)

            if sel_sortset_lte is not None and sel_sortset is not None:
                raise ValueError(
                    f"Cannot specify both sortset and sortset_lte in selection"
                )

            sel_output = selection.get("output", None)

            # First, we get the sortset-level information.
            if sel_sortset is not None:
                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] == sel_sortset
                ]

            if sel_sortset_lte is not None:
                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] <= sel_sortset_lte
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

            # Similarly, select for certain outputs
            if sel_output is not None:
                sessions = [
                    session
                    for session in sessions
                    if sel_output in session["fields"].keys()
                ]

            session_names += [session["id"] for session in sessions]

            # Now we get the chunk-level information.
            max_chunks = selection.get("max_examples", 1e12)
            for session in sessions:
                num_chunks = 0
                for trial in session["trials"]:
                    for chunk in trial["chunks"].get(self.split, []):
                        iomap = {
                            k: session[k]
                            for k in ["fields", "task"]
                        }

                        # Check that the chunk has the requisite inputs.
                        check = check_include(included_datasets, iomap["fields"])
                        if not check:
                            continue

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

                        num_chunks += 1
                        if num_chunks >= max_chunks:
                            print(f"Truncating at {num_chunks} examples")
                            break
                    if num_chunks >= max_chunks:
                        break

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

        if not hasattr(data.spikes, "unit_index"):
            warnings.warn(
                "Unit index is not set, backing out from sortset. This is slow. You should set the unit index in prepare_data.py files for speed. In a future version, this will turn into an error."
            )

            # Inverse map from unit name to index.
            inv_map = {k: v for v, k in enumerate(data.units.unit_name)}
            data.spikes.unit_index = torch.Tensor(
                [inv_map[x] for x in data.spikes.names]
            ).to(dtype=torch.long)

        if hasattr(data.spikes, "names"):
            delattr(data.spikes, "names")

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
    output_timestamps: TensorType["batch", "max_ntout"]
    output_task_indices: Dict[str, TensorType["*ntout_task", 2, torch.int32]]
    output_values: Dict[str, TensorType["*ntout_task", "*nchannelsout"]]
    output_weights: Dict[str, TensorType["*ntout_task"]]

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


def check_include(description: Dict, keys: Dict) -> bool:
    if "include_input" in description:
        return all([key in keys for key in description["include_input"]])
    else:
        return True


def check_include_exclude(description, key):
    if "include_input" in description:
        return key in description["include_input"]
    elif "exclude_input" in description.keys():
        return key not in description["exclude_input"]
    else:
        return True


def resolve(data, key) -> torch.Tensor:
    # Split key by dots, resolve using getattr
    components = key.split(".")
    for c in components:
        try:
            data = getattr(data, c)
        except AttributeError:
            raise AttributeError(
                f"Could not resolve {key} in data (specifically, at level {c}))"
            )
    return data


TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "long": torch.long,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class Collate:
    def __init__(
        self,
        num_latents_per_step: int = 1,
        step=1.0,
        reweight: bool = False,
        sequence_length=1.0,
        unit_vocab: Optional[torchtext.vocab.Vocab] = None,
        decoder_registry: Optional[Dict[str, DecoderSpec]] = None,
        weight_registry: Optional[Dict[int, float]] = None,
    ):
        """Stack datasets into a batch.

        Note that there are necessarily sequence_length / step * num_latents_per_step latent tokens.
        """
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.reweight = reweight
        self.unit_vocab = unit_vocab

        # TODO: remove sequence_length from the parameters and read the sequence length
        # from the data instead.
        self.sequence_length = sequence_length
        if decoder_registry is None:
            raise NotImplementedError("Decoder registry is required")
        self.decoder_registry = decoder_registry
        if weight_registry is None:
            weight_registry = {}
        self.weight_registry = weight_registry

    def __call__(self, batch: List[Data]) -> Dict[str, Union[torch.Tensor, List]]:
        # Make sure that the unit vocab has a mapping for NA and that it corresponds to 0
        assert self.unit_vocab.forward(["NA"])[0] == 0

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

        # max_output_tokens = collections.defaultdict(int)

        # Measures max output timestamps in 1 sample
        num_max_output_timestamps = 0
        # Measures total number of outputs for each task
        num_outputs_taskwise = collections.defaultdict(lambda: 0)
        # Set of all registry keys we see in this batch
        decoder_registry_keys = set()

        for i, data in enumerate(batch):
            # Two conditions: the data exists, and it's been requested.
            data.has_spike_waveforms = False
            data.has_average_waveforms = False
            data.has_lfps = False
            if (str(RecordingTech.UTAH_ARRAY_WAVEFORMS)) in data.iomap["fields"].keys():
                check = check_include_exclude(
                    data.description, str(RecordingTech.UTAH_ARRAY_WAVEFORMS)
                )
                if check:
                    max_spike_waveform_size = max(
                        max_spike_waveform_size, data.spikes.waveforms.shape[1]
                    )
                    data.has_spike_waveforms = check
                    has_spike_waveforms = check

            if (str(RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS)) in data.iomap[
                "fields"
            ].keys():
                check = check_include_exclude(
                    data.description,
                    str(RecordingTech.UTAH_ARRAY_AVERAGE_WAVEFORMS),
                )
                if check:
                    max_average_waveform_size = max(
                        max_average_waveform_size,
                        data.units.average_waveform.shape[1],
                    )
                    data.has_average_waveforms = check
                    has_average_waveforms = check

            if (str(RecordingTech.UTAH_ARRAY_LFPS)) in data.iomap["fields"].keys():
                check = check_include_exclude(
                    data.description, str(RecordingTech.UTAH_ARRAY_LFPS)
                )
                if check:
                    max_nbands = max(max_nbands, len(data.lfp_metadata.bands))
                    data.has_lfps = check
                    has_lfps = check

            # Now we deal with the outputs.
            num_output_timestamps = (
                0  # measures number of output timestamps for this sequence sample
            )
            for metric in data.description["metrics"]:
                key = metric["output_key"]
                decoder_registry_keys.add(key)
                value = resolve(data, self.decoder_registry[key].value_key)
                num_output_timestamps += value.shape[0]
                num_outputs_taskwise[key] += value.shape[0]
            num_max_output_timestamps = max(
                num_max_output_timestamps, num_output_timestamps
            )

        if has_spike_waveforms:
            spike_waveforms = torch.zeros(
                len(batch), max_num_tokens, max_spike_waveform_size
            )

        if has_average_waveforms:
            average_waveforms = torch.zeros(max_num_units, max_average_waveform_size)

        if has_lfps:
            lfps = torch.zeros(
                (len(batch), max_num_tokens, max_nbands), dtype=torch.float32
            )

        # Initialize output tensors
        output_timestamps = torch.zeros(len(batch), num_max_output_timestamps)
        output_task_indices = {}
        output_values = {}
        output_weights = {}
        output_offset = {}
        for key in decoder_registry_keys:
            reg = self.decoder_registry[key]
            dim = reg.target_dim
            dtype = TORCH_DTYPES[reg.target_dtype]
            output_task_indices[key] = torch.zeros(
                num_outputs_taskwise[key], 2, dtype=torch.int32
            )
            output_values[key] = torch.zeros(num_outputs_taskwise[key], dim).to(
                dtype=dtype
            )
            output_weights[key] = torch.zeros(num_outputs_taskwise[key])
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

        # fill values
        for i, data in enumerate(batch):
            # add spike events
            spikes = data.spikes

            # Do a lookup from unit names to their location in the embedding
            unit_names = data.units.unit_name
            if isinstance(unit_names, np.ndarray):
                unit_names = unit_names.tolist()
            unit_embedding_index = torch.tensor(self.unit_vocab.forward(unit_names))
            mapped_spikes = torch.cat(
                (
                    unit_embedding_index[spikes.unit_index],
                    unit_embedding_index,
                    unit_embedding_index,
                )
            )

            spike_ids[i, : len(mapped_spikes)] = mapped_spikes
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
                spike_waveforms[
                    i, : len(spikes), : spikes.waveforms.shape[1]
                ] = spikes.waveforms

            # Add average waveforms
            if data.has_average_waveforms:
                inverse_slot = self.unit_vocab(units.tolist())
                average_waveforms[
                    inverse_slot, : data.units.average_waveform.shape[1]
                ] = torch.Tensor(data.units.average_waveform)

            # Add local field potentials
            if data.has_lfps:
                # We do a local, nearest neighbour lookup to find the LFPs corresponding to each spike.
                # We use the LFP index to do so.
                # Build a map from unit name to LFP index.
                lfp_channel_to_idx = {
                    x: i for i, x in enumerate(data.lfp_metadata.channels)
                }
                unit_to_channel = torch.tensor(
                    [lfp_channel_to_idx[c] for c in data.units.channel_name]
                )
                lfp_idx = unit_to_channel[spikes.unit_index]

                # Now find the corresponding offset using nearest neighbour.
                # Note that this is only safe to do on a RegularTimeSeries
                assert isinstance(data.lfps, RegularTimeSeries)
                lfp_approx_index = data.lfps.sampling_rate * (
                    spikes.timestamps - data.lfps.timestamps[0]
                )
                lfp_tidx = torch.clip(
                    torch.round(lfp_approx_index),
                    0,
                    data.lfps.lfp.shape[0] - 1,
                ).to(torch.long)
                assert len(lfp_idx) == len(lfp_tidx)
                selected_lfp = data.lfps.lfp[lfp_tidx, lfp_idx, :]

                lfps[i, : selected_lfp.shape[0], : selected_lfp.shape[1]] = selected_lfp

            # Now we deal with the outputs.
            timestamps_offset = 0
            for metric in data.description["metrics"]:
                key = metric["output_key"]
                timestamps = resolve(data, self.decoder_registry[key].timestamp_key)
                values = resolve(data, self.decoder_registry[key].value_key)
                num_outputs = timestamps.shape[0]

                # Output timestamps are assigned in the standard batch-index-wise manner
                timestamps_range = timestamps_offset + torch.arange(
                    0, num_outputs, dtype=torch.int32
                )
                output_timestamps[
                    i, timestamps_range
                ] = timestamps.float()  # WARNING! Timestamps precision reduction here

                # Other output things are assigned in a task-wise manner
                offset = output_offset[key]
                output_task_indices[key][
                    offset : offset + num_outputs, 0
                ] = i  # batch-index
                output_task_indices[key][
                    offset : offset + num_outputs, 1
                ] = timestamps_range  # sequence-index
                output_values[key][offset : offset + num_outputs, :] = values

                try:
                    behavior_type = resolve(
                        data, self.decoder_registry[key].behavior_type_key
                    )
                except AttributeError:
                    behavior_type = torch.zeros(num_outputs, dtype=torch.long)
                found = [
                    self.weight_registry.get(int(x.item()), False)
                    for x in behavior_type
                ]
                if not all(found) and any(found):
                    idx = np.where(np.array(found) == False)[0]
                    raise ValueError(
                        f"Could not find weights for behavior #{behavior_type[idx]}"
                    )
                weights = [
                    self.weight_registry.get(int(x.item()), 1.0) for x in behavior_type
                ]
                output_weights[key][offset : offset + num_outputs] = torch.tensor(
                    weights
                ) * metric.get("weight", 1.0)

                timestamps_offset += num_outputs
                output_offset[key] += num_outputs

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
            output_task_indices=output_task_indices,
            output_timestamps=output_timestamps,
            output_values=output_values,
            output_weights=output_weights,
            session_names=session_names,
            **extras,
        )
        return dataclasses.asdict(data)


def build_vocab(
    train_units: List[str],
    val_units: Optional[List[str]] = None,
    test_units: Optional[List[str]] = None,
) -> torchtext.vocab.Vocab:
    """
    Build a vocabulary from a list of unit names. This is used to map unit
    names to indices in the collator.

    Args:
        train_units: List of unit names in the training dataset.
        val_units: List of unit names in the validation dataset.
        test_units: List of unit names in the test dataset.

    Returns:
        A torchtext.vocab.Vocab object.
    """
    # Check that val dataset unit names overlap train dataset unit names
    if val_units is not None:
        assert set(val_units).issubset(
            set(train_units)
        ), "Validation dataset units must be a subset of train dataset"
    # Same for test
    if test_units is not None:
        assert set(test_units).issubset(
            set(train_units)
        ), "Validation dataset units must be a subset of train dataset"

    unit_names = train_units
    if val_units is not None:
        unit_names += val_units
    if test_units is not None:
        unit_names += test_units

    unit_names = sorted(list(set(list(unit_names))))
    od = collections.OrderedDict({x: 1 for x in unit_names})
    vocab = torchtext.vocab.vocab(od, specials=["NA"])
    return vocab
