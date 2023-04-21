import os
import re
from collections import OrderedDict

import torch
from einops import repeat


class Dataset(torch.utils.data.Dataset):
    extension = '.pt'
    pattern = re.compile(r'^([\w\d]+)_\d+\.pt$')

    def __init__(self, root, split, include=None, transform=None):
        super().__init__()
        self.root = root

        assert split in ['train', 'valid', 'test']
        self.split = split

        assert include is not None, 'Please specify the datasets to include'
        if isinstance(include, str):
            include = [include]
        self.include = [self.__parse_include_arg(include_dir) for include_dir in include]
        
        self.session_ptr = OrderedDict()
        self.total_num_units = 0

        self.transform = transform
        self.filenames, self.session_ids = self.look_for_files()

        self.session_id_tokens = dict(zip(self.session_ptr.keys(), range(len(self.session_ptr))))

    def __parse_include_arg(self, include):
        session = None
        dataset_dir, session_list = include.split('/')
        if session_list == '*':
            session_list = 'all'
        
        if os.path.exists(os.path.join(self.root, dataset_dir, session_list + '.txt')):
            session_list_filename = session_list + '.txt'
        else:
            session = session_list
            session_list = 'all'
        session_list_filename = session_list + '.txt'
        return dataset_dir, session_list_filename, session

    def look_for_files(self):
        files = []
        file_session_ids = []

        for include_dir, include_file_list, only_session in self.include:
            session_ids = self.__parse_file_list(os.path.join(self.root, include_dir, include_file_list), only_get=only_session)
            include_dir = os.path.join(self.root, include_dir, 'processed', self.split)

            for file in sorted(os.listdir(include_dir)):
                if file.endswith(self.extension):
                    session_id = self.parse_session_id(file)
                    if session_id in session_ids:
                        files.append(os.path.join(include_dir, file))
                        file_session_ids.append(session_id)
        return files, file_session_ids

 
    def register_new_session(self, session_id, num_units):
        if session_id in self.session_ptr:
            raise ValueError(f'Session {session_id} already registered, duplicate session files are not allowed.')
        self.session_ptr[session_id] = (self.total_num_units, self.total_num_units + num_units - 1)
        self.total_num_units += num_units

    def __parse_file_list(self, path, only_get=None):
        session_ids = []
        only_get_session_found = False
        with open(path, 'r') as f:
            for line in f.readlines():
                session_id, num_units = line.strip().split(' ')
                if only_get is not None:
                    if session_id == only_get:
                        self.register_new_session(session_id, int(num_units))
                        session_ids.append(session_id)
                        only_get_session_found = True
                        break
                else:
                    self.register_new_session(session_id, int(num_units))
                    session_ids.append(session_id)
        if only_get is not None and not only_get_session_found:
            raise ValueError(f'Could not find session {only_get} in file list {path}')
        return session_ids

    def parse_session_id(self, filename):
        filename = os.path.basename(filename)
        match = self.pattern.match(filename)
        if match:
            extracted_session = match.group(1)
            return extracted_session
        else:
            raise ValueError(f'Could not parse session id from filename {filename}')

    def __getitem__(self, item):
        data = torch.load(self.filenames[item])
        # translate unit ids
        session_id = self.session_ids[item]
        translate = self.session_ptr[session_id][0]
        data.spikes.unit_id += translate
        data.units.id += translate
        data.session_id = session_id
        data.session_id_token = self.session_id_tokens[session_id]

        # apply transform
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.filenames)

    def few_shot(self, num_samples, shuffle=True):
        assert num_samples <= len(self), f'Cannot sample {num_samples} from dataset of length {len(self)}'
        if shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = torch.arange(len(self))
        self.filenames = [self.filenames[i] for i in indices[:num_samples]]
        return self


def next_multiple_of_8(x):
    remainder = x % 8
    if remainder == 0:
        return x
    else:
        return x + (8 - remainder)


class Collate:
    def __init__(self, max_num_units=4096, num_latents_per_step=1, step=1.0,
                 behavior_type_weight=None, reweight=False, sequence_length=1.0):
        self.max_num_units = max_num_units + 1
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.behavior_type_weight = behavior_type_weight
        self.reweight = reweight
        self.sequence_length = sequence_length

    def __call__(self, batch):
        # make spike tensors
        num_tokens = [len(data.spikes) + len(data.units.id) * 2 for data in batch]
        max_num_tokens = next_multiple_of_8(max(num_tokens))

        spike_timestamps = torch.zeros((len(batch), max_num_tokens), dtype=torch.float32)
        spike_unit = torch.empty((len(batch), max_num_tokens), dtype=torch.long).fill_(self.max_num_units-1)
        spike_type = torch.zeros((len(batch), max_num_tokens), dtype=torch.long)
        mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)

        num_output_tokens = [len(data.behavior.timestamps) for data in batch]
        max_num_output_tokens = next_multiple_of_8(max(num_output_tokens))

        # make behavior tensors
        output_timestamps = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.float32)
        output_values = torch.empty((len(batch), max_num_output_tokens, 2), dtype=torch.float32).fill_(1e6)
        output_weight = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.float32)
        output_stage = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.long)

        # make latent tensors
        latent_timestamps = torch.arange(0, self.sequence_length, self.step) + self.step / 2
        latent_ids = torch.arange(self.num_latents_per_step, dtype=torch.long)
        num_timestamps = len(latent_timestamps)
        latent_timestamps = repeat(latent_timestamps, 't -> b (t u)', b=len(batch), u=len(latent_ids))
        latent_ids = repeat(latent_ids, 'u -> b (t u)', b=len(batch), t=num_timestamps)

        num_timestamps = latent_timestamps.size(1)

        # make attn masks
        input_mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)
        output_mask = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.bool)

        # fill values
        for i, data in enumerate(batch):
            # add spike events
            spikes = data.spikes
            spike_timestamps[i, :len(spikes)] = spikes.timestamps
            spike_unit[i, :len(spikes)] = spikes.unit_id
            mask[i, :len(spikes)] = True
            # add artificial start and end of trial events to each unit
            units = data.units.id
            start, end = data.start, data.end
            # assume that aligned with start and end
            start, end = 0., end - start
            spike_timestamps[i, len(spikes):len(spikes) + len(units)] = start
            spike_timestamps[i, len(spikes) + len(units):len(spikes) + len(units) * 2] = end
            spike_unit[i, len(spikes):len(spikes) + len(units)] = units
            spike_unit[i, len(spikes) + len(units):len(spikes) + len(units) * 2] = units
            spike_type[i, len(spikes):len(spikes) + len(units)] = 1
            spike_type[i, len(spikes) + len(units):len(spikes) + len(units) * 2] = 2
            
            # make output
            output = data.behavior
            output_timestamps[i, :len(output.timestamps)] = output.timestamps
            output_values[i, :len(output.timestamps)] = output.hand_vel
            output_mask[i, :len(output.timestamps)] = True

            behavior_type = output.type if hasattr(output, 'type') else output.behavior_type
            output_stage[i, :len(output.timestamps)] = behavior_type
            output_weight[i, :len(output.timestamps)] = self.behavior_type_weight[behavior_type] if self.behavior_type_weight is not None else 1.0
            # reweight so that each trial is equally important
            if self.reweight:
                output_weight[i] *= max_num_output_tokens / len(output.timestamps)

            # update masks
            input_mask[i, :len(spikes) + len(units) *2] = True

        # session id
        session_id = [data.session_id for data in batch]
        task_id = torch.tensor([data.session_id_token for data in batch],  dtype=torch.long)

        task_id = repeat(task_id, 'b -> b t', t=max_num_output_tokens)

        data = dict(
            spike_timestamps=spike_timestamps,
            spike_unit=spike_unit,
            spike_type=spike_type,
            mask=mask,
            output_timestamps=output_timestamps,
            output_values=output_values,
            output_weight=output_weight,
            output_mask=output_mask,
            output_stage=output_stage,
            task_id=task_id,
            latent_timestamps=latent_timestamps,
            latent_id=latent_ids,
            input_mask=input_mask,
            session_id=session_id,
        )
        return data


def prepare_sample(data, step=1./8, num_latents_per_step=16):
    # prepare input
    num_input_tokens = len(data.spikes) + len(data.units.id) * 2
    spike_timestamps = torch.zeros((num_input_tokens), dtype=torch.float32)
    spike_unit = torch.zeros((num_input_tokens), dtype=torch.long)
    spike_type = torch.zeros((num_input_tokens), dtype=torch.long)
    spikes = data.spikes
    spike_timestamps[:len(data.spikes)] = data.spikes.timestamps
    spike_unit[:len(data.spikes)] = data.spikes.unit_id
    units = data.units.id
    start, end = data.start, data.end
    start, end = 0., end - start
    spike_timestamps[len(spikes):len(spikes) + len(units)] = start
    spike_timestamps[len(spikes) + len(units):len(spikes) + len(units) * 2] = end
    spike_unit[len(spikes):len(spikes) + len(units)] = units
    spike_unit[len(spikes) + len(units):len(spikes) + len(units) * 2] = units
    spike_type[len(spikes):len(spikes) + len(units)] = 1
    spike_type[len(spikes) + len(units):len(spikes) + len(units) * 2] = 2

    # prepare output
    output_timestamps = data.behavior.timestamps
    output_values = data.behavior.hand_vel
    output_stage = data.behavior.type if hasattr(data.behavior, 'type') else data.behavior.behavior_type

    # prepare latent
    sequence_length = data.end - data.start
    latent_timestamps = torch.arange(0, sequence_length, step) + step / 2
    latent_ids = torch.arange(num_latents_per_step, dtype=torch.long)
    num_timestamps = len(latent_timestamps)
    latent_timestamps = repeat(latent_timestamps, 't -> (t u)', u=len(latent_ids))
    latent_ids = repeat(latent_ids, 'u -> (t u)', t=num_timestamps)

    # session id
    session_id = data.session_id
    task_id = torch.ones(len(output_timestamps),  dtype=torch.long) * data.session_id_token


    data = dict(
        spike_timestamps=spike_timestamps,
        spike_unit=spike_unit,
        spike_type=spike_type,
        mask=None,
        output_timestamps=output_timestamps,
        output_values=output_values,
        output_weight=None,
        output_mask=None,
        output_stage=output_stage,
        task_id=task_id,
        latent_timestamps=latent_timestamps,
        latent_id=latent_ids,
        input_mask=None,
        session_id=session_id,
        )
    return data
