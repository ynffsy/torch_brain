import numpy as np
import torch

from kirby.utils import logging

log = logging(header="UNIT DROPOUT", header_color="cyan")

"""
Triangular distribution with a peak at mode_units, going from min_units to max_units.
"""


class UnitCustomDistribution:
    def __init__(
        self,
        min_units=20,
        mode_units=100,
        max_units=300,
        peak=4,
        M=10,
        max_attempts=100,
        seed=None,
    ):
        super().__init__()
        self.min_units = min_units
        self.mode_units = mode_units
        self.max_units = max_units
        self.peak = peak
        self.M = M
        self.max_attempts = max_attempts

        self.rng = np.random.default_rng(seed=seed)

    def unnormalized_density_function(self, x):
        if x < self.min_units:
            return 0
        if x <= self.mode_units:
            return 1 + (self.peak - 1) * (x - self.min_units) / (
                self.mode_units - self.min_units
            )
        if x <= self.max_units:
            return self.peak - (self.peak - 1) * (x - self.mode_units) / (
                self.max_units - self.mode_units
            )
        return 1

    def proposal_distribution(self, x):
        return self.rng.uniform()

    def sample(self, num_units):
        if num_units < self.min_units:
            # log.warning(f"Requested {num_units} units, but minimum is {self.min_units}")
            return num_units

        # uses rejection sampling
        num_attempts = 0
        while True:
            x = self.min_units + self.rng.uniform() * (
                self.max_units - self.min_units
            )  # Sample from the proposal distribution
            u = self.rng.uniform()
            if u <= self.unnormalized_density_function(x) / (
                self.M * self.proposal_distribution(x)
            ):
                return x
            num_attempts += 1
            if num_attempts > self.max_attempts:
                # warning
                log.warning(
                    f"Could not sample from distribution after {num_attempts} attempts, using all units"
                )
                return num_units


class UnitDropout:
    r"""Augmentation that randomly drops units from the data object.
    
    ..note:: 
        Use `UnitDropout` before `RandomCrop` would be more efficient if index_maps are 
        precomputed. 
    """
    def __init__(self, *args, **kwargs):
        self.distribution = UnitCustomDistribution(*args, **kwargs)

    def __call__(self, data):
        # get units
        unit_names = data.units.unit_name
        num_units = len(unit_names)

        num_units_to_sample = int(self.distribution.sample(num_units))

        # shuffle units and take the first num_units_to_sample
        keep_indices, _ = torch.sort(torch.randperm(num_units)[:num_units_to_sample])

        # update the units
        units_keep = data.units.__class__.__new__(data.units.__class__)
        for key, value in data.units.__dict__.items():
            if value is not None and isinstance(value, torch.Tensor):
                units_keep.__dict__[key] = value[keep_indices]
            elif value is not None and isinstance(value, np.ndarray):
                units_keep.__dict__[key] = value[keep_indices.numpy()]
            elif value is not None and isinstance(value, list):
                units_keep.__dict__[key] = [value[x.item()] for x in keep_indices]
            else:
                units_keep.__dict__[key] = None
        
        data.units = units_keep

        # Keep only the relevant spikes
        if hasattr(data.spikes, 'names_index_dict'):
            # index maps have been precomputed, this should make things way faster! 
            spikes_keep_indices = []
            for name in data.units.unit_name:
                if name in data.spikes.names_index_dict:
                    spikes_keep_indices.append(data.spikes.names_index_dict[name]) 
            spikes_keep_indices = torch.concatenate(spikes_keep_indices)
            # sort
            spikes_keep_indices = spikes_keep_indices.sort()[0]
            
            # update the spikes
            spikes_keep = data.spikes.__class__.__new__(data.spikes.__class__)
            for key, value in data.spikes.__dict__.items():
                if value is not None and isinstance(value, torch.Tensor):
                    spikes_keep.__dict__[key] = value[spikes_keep_indices]
                elif value is not None and isinstance(value, np.ndarray):
                    spikes_keep.__dict__[key] = value[spikes_keep_indices.numpy()]
                elif value is not None and isinstance(value, list):
                    spikes_keep.__dict__[key] = [value[x.item()] for x in spikes_keep_indices]
                else:
                    spikes_keep.__dict__[key] = None
            
            delattr(data.spikes, 'names_index_dict')
            spikes_keep._sorted = data.spikes.sorted
            data.spikes = spikes_keep
        else:
            # todo add warning when index maps are not precomputed
            total_spike_mask = torch.zeros_like(data.spikes.timestamps, dtype=torch.bool)
            n = np.array(data.spikes.names)
            for unit_name in data.units.unit_name:
                total_spike_mask |= n == unit_name

            data.spikes.timestamps = data.spikes.timestamps[total_spike_mask == 1]
            data.spikes.types = data.spikes.types[total_spike_mask == 1]
            # Tricky bit: torch doesn't have real booleans, only uint8, so when we convert 
            # back to numpy we have to cast correctly to obtains the desired effect.
            data.spikes.names = (n[total_spike_mask.cpu().detach().numpy() == 1]).tolist()

        return data
