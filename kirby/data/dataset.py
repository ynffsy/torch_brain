import collections
import dataclasses
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import msgpack
import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchtext
from einops import repeat
from torchtyping import TensorType

from kirby.data import Data
from kirby.data.data import Interval, RegularTimeSeries
import kirby.taxonomy
from kirby.taxonomy import StringIntEnum, description_helper
from kirby.taxonomy.taxonomy import DecoderSpec, RecordingTech


@dataclass
class SessionFileInfo:
    """Information about a session that is pertinent to the dataset object.
    Its goal is to be able to load the session file and extract the relevant
    data from it.
    """

    session_id: str  # <dandiset>/<session_id>, fully qualified, should be unique
    filename: Path
    iomap: Dict[str, Any]
    description: Dict[str, Any]
    sampling_interval: Interval  # Intervals to sample from


@dataclass
class DatasetIndex:
    """Information needed to extract a slice from a dataset."""

    session_id: str
    start: float
    end: float


class Dataset(torch.utils.data.Dataset):
    r"""This class abstracts a collection of lazily-loaded Data objects. Each of these
    Data objects corresponds to a session and lives on the disk until it is requested.
    The `include` argument guides which sessions are included in this Dataset.
    To request a piece of a included session's data, you can use the `get` method,
    or index the Dataset with a `DatasetIndex` object (see `__getitem__`).

    This definition is a deviation from the standard PyTorch Dataset definition, which
    generally presents the dataset directly as samples. In this case, the Dataset
    by itself does not provide you with samples, but rather the means to flexibly work
    and accesss complete sessions.
    Within this framework, it is the job of the sampler to provide the
    DatasetIndex indices to slice the dataset into samples (see `kirby.data.sampler`).
    """

    _check_for_data_leakage_flag: bool = True
    _open_files: Optional[Dict[str, h5py.File]] = None
    _data_objects: Optional[Dict[str, Data]] = None

    def __init__(
        self,
        root: str,
        split: str,
        include: List[Dict[str, Any]],
        transform=None,
        keep_files_open: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split

        if include is None:
            raise ValueError("Please specify the datasets to include")

        self.include = include
        self.transform = transform
        self.session_info_dict, self.unit_ids = self._look_for_files()
        self.session_ids: List[str] = [
            x.session_id for x in self.session_info_dict.values()
        ]

        if keep_files_open:
            self._open_files = {
                session_id: h5py.File(session_info.filename, "r")
                for session_id, session_info in self.session_info_dict.items()
            }

            self._data_objects = {
                session_id: Data.from_hdf5(f)
                for session_id, f in self._open_files.items()
            }

        self.requested_keys = None
    
    def _close_open_files(self):
        """Closes the open files and deletes open data objects. 
        This is useful when you are done with the dataset.
        """
        if self._open_files is not None:
            for f in self._open_files.values():
                f.close()
            self._open_files = None

        self._data_objects = None # initialized Data objects should be gc'd 

    def __del__(self):
        self._close_open_files()

    def _look_for_files(self) -> Tuple[Dict[str, SessionFileInfo], List[str]]:
        session_ids = []
        unit_ids = []
        session_info_dict = {}

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
            all_sortset_ids = [x["id"] for x in sortsets]
            all_sortset_subjects = set([x["subject"] for x in sortsets])

            # Perform selection. Right now, we are limiting ourselves to sortset,
            # subject and session, but we could make selection more flexible in the
            # future.
            sel_sortset = selection.get("sortset", None)
            sel_sortsets = selection.get("sortsets", None)
            sel_sortset_lte = selection.get("sortset_lte", None)
            sel_subject = selection.get("subject", None)
            sel_subjects = selection.get("subjects", None)
            # exclude_sortsets allows you to exclude some sortsets from the selection.
            # example use: you want to train on the complete dandiset, but leave out
            # a few sortsets for evaluating transfer performance.
            sel_exclude_sortsets = selection.get("exclude_sortsets", None)

            sel_session = selection.get("session", None)
            sel_output = selection.get("output", None)

            filtered = False
            if sel_sortset is not None:
                assert (
                    sel_sortset in all_sortset_ids
                ), f"Sortset {sel_sortset} not found in dandiset {selection['dandiset']}"
                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] == sel_sortset
                ]
                filtered = True

            if sel_sortsets is not None:
                assert not filtered, "Cannot specify sortset AND sortsets in selection"

                # Check that all sortsets are in the dandiset.
                for sortset in sel_sortsets:
                    assert (
                        sortset in all_sortset_ids
                    ), f"Sortset {sortset} not found in dandiset {selection['dandiset']}"

                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] in sel_sortsets
                ]
                filtered = True

            if sel_sortset_lte is not None:
                assert (
                    not filtered
                ), "Cannot specify sortset_lte AND sortset(s) in selection"

                sortsets = [
                    sortset for sortset in sortsets if sortset["id"] <= sel_sortset_lte
                ]
                filtered = True

            if sel_subject is not None:
                assert (
                    not filtered
                ), "Cannot specify subject AND sortset(s)/sortset_lte in selection"

                assert (
                    sel_subject in all_sortset_subjects
                ), f"Could not find subject {sel_subject} in dandiset {selection['dandiset']}"

                sortsets = [
                    sortset for sortset in sortsets if sortset["subject"] == sel_subject
                ]
                filtered = True

            if sel_subjects is not None:
                assert (
                    not filtered
                ), "Cannot specify subjects AND subject/sortset(s)/sortset_lte in selection"

                # Make sure all subjects asked for are in the dandiset
                sel_subjects = set(sel_subjects)
                assert sel_subjects.issubset(all_sortset_subjects), (
                    f"Could not find subject(s) {sel_subjects - all_sortset_subjects} "
                    f" in dandiset {selection['dandiset']}"
                )

                sortsets = [
                    sortset
                    for sortset in sortsets
                    if sortset["subject"] in sel_subjects
                ]
                filtered = True

            # Exclude sortsets if asked.
            if sel_exclude_sortsets is not None:
                sortsets = [
                    sortset
                    for sortset in sortsets
                    if sortset["id"] not in sel_exclude_sortsets
                ]

            # Note that this logic may result in adding too many slots but that's fine.
            unit_ids += [x for sortset in sortsets for x in sortset["units"]]
            # unit_ids are already fully qualified with prepended dandiset id.

            # Now we get the session-level information.
            sessions = sum([sortset["sessions"] for sortset in sortsets], [])
            if sel_session is not None:
                sessions = [
                    session for session in sessions if session["id"] == sel_session
                ]

            assert len(sessions) > 0, f"No sessions found for {i}'th selection included"

            # Similarly, select for certain outputs
            if sel_output is not None:
                sessions = [
                    session
                    for session in sessions
                    if sel_output in session["fields"].keys()
                ]

            session_ids += [session["id"] for session in sessions]

            # Now we get the session-level information
            for session in sessions:
                iomap = {k: session[k] for k in ["fields", "task"]}

                # Check that the chunk has the requisite inputs.
                check = check_include(included_datasets, iomap["fields"])
                if not check:
                    continue

                session_id = selection["dandiset"] + "/" + session["id"]
                session_info_dict[session_id] = SessionFileInfo(
                    session_id=session_id,
                    filename=(Path(self.root) / (session_id + ".h5")),
                    iomap=iomap,
                    description=included_datasets,
                    sampling_interval=Interval.from_list(session["splits"][self.split]),
                )

        all_filenames = [x.filename for _, x in session_info_dict.items()]
        assert len(set(all_filenames)) == len(
            all_filenames
        ), f"All selected filenames should be unique"

        unit_ids = list(set(unit_ids))
        return session_info_dict, unit_ids

    def request_keys(self, request_keys):
        self.requested_keys = request_keys

    def get_interval_dict(self):
        """Returns a dictionary of interval-list for each session.
        Each interval-list is a list of tuples (start, end) for each interval.
        """
        intervals = {}
        for session_id, session_info in self.session_info_dict.items():
            intervals[session_id] = list(
                zip(
                    session_info.sampling_interval.start,
                    session_info.sampling_interval.end,
                )
            )
        return intervals

    def get(self, session_id: str, start: float, end: float):
        """Get a slice of the dataset.
        Args:
            session_id: The session id of the slice. Note this is the fully qualified
                session-id: <dandiset>/<session_id>
            start: The start time of the slice.
            end: The end time of the slice.
        """
        session_info = self.session_info_dict[session_id]
        if self._data_objects is None:
            filename = session_info.filename
            with h5py.File(filename, "r") as f:
                data = Data.from_hdf5(f)
                sample = data.slice(start, end, request_keys=self.requested_keys)
        else:
            data = self._data_objects[session_id]
            sample = data.slice(start, end, request_keys=self.requested_keys)

        if self._check_for_data_leakage_flag:
            sample._check_for_data_leakage(self.split)

        sample.session = session_id
        sample.description = session_info.description
        sample.iomap = session_info.iomap
        return sample

    def disable_data_leakage_check(self):
        self._check_for_data_leakage_flag = False
        logging.warn(
            f"Data leakage check is disabled. Please be absolutely sure that there is "
            f"no leakage between {self.split} and other splits (eg. the test split)."
        )

    def __getitem__(self, index: DatasetIndex):
        sample = self.get(index.session_id, index.start, index.end)

        # apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        raise NotImplementedError("Length of dataset is not defined")

    def __iter__(self):
        raise NotImplementedError("Iteration over dataset is not defined")


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
    elif "exclude_input" in description:
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

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
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
        decoder_registry: Optional[
            Dict[str, DecoderSpec]
        ] = kirby.taxonomy.decoder_registry,
        weight_registry: Optional[Dict[int, float]] = kirby.taxonomy.weight_registry,
        metrics: Optional[List[Dict[str, str]]] = None,
    ):
        """Stack datasets into a batch.

        Note that there are necessarily sequence_length / step * num_latents_per_step latent tokens.

        Args:
            metrics: A list of metrics dictionary including output keys and weight.
                If None, metrics will be inferred from the dataset.
                Example: metrics=[{"output_key": "CURSORVELOCITY2D", "weight": 1.0}]
        """
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.reweight = reweight
        self.unit_vocab = unit_vocab
        self.metrics = metrics

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
            metrics = self.metrics or data.description["metrics"]
            for metric in metrics:
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
            spike_timestamps[i, : len(spikes)] = torch.from_numpy(spikes.timestamps)
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
            metrics = self.metrics or data.description["metrics"]
            for metric in metrics:
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

                weights = torch.tensor(
                    [
                        self.weight_registry.get(int(x.item()), -1.0)
                        for x in behavior_type
                    ]
                )

                # Either we have weights for all or for none (implicitly, everything
                # has a weight of 1 in that case). There shouldn't be any
                # in-between cases, which would mean there's an undefined behaviour.
                if torch.any(weights == -1.0) and not torch.all(weights == -1.0):
                    idx = torch.where(weights == 0)[0][0]
                    raise ValueError(
                        f"Could not find weights for behavior #{behavior_type[idx]}"
                    )

                weights[weights == -1.0] = 1.0

                output_weights[key][
                    offset : offset + num_outputs
                ] = weights * metric.get("weight", 1.0)

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

        data = dict(
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
        return data


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
