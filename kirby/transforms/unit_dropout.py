import logging
import numpy as np

from kirby.data import Data, RegularTimeSeries, IrregularTimeSeries


class TriangleDistribution:
    r"""Triangular distribution with a peak at mode_units, going from min_units to
    max_units.
    """
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

        # TODO pass a generator?
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
                logging.warning(
                    f"Could not sample from distribution after {num_attempts} attempts,"
                     " using all units."
                )
                return num_units


class UnitDropout:
    r"""Augmentation that randomly drops units from the sample. This will remove the
    dropped units from `data.units` and all spikes from the dropped units. This will
    also relabel the unit_index in `data.spikes`.

    This currently assumes that the data object contains `data.units.id` and
    `data.spikes.unit_index`.
    """
    def __init__(self, field="spikes", *args, **kwargs):
        # TODO allow multiple fields (example: spikes + LFP)
        self.field = field
        # TODO this currently assumes the type of distribution we use, in the future,
        # the distribution might be passed as an argument.
        self.distribution = TriangleDistribution(*args, **kwargs)

    def __call__(self, data: Data):
        # get units from data
        unit_ids = data.units.id
        num_units = len(unit_ids)

        # sample the number of units to keep from the population
        num_units_to_sample = int(self.distribution.sample(num_units))

        # shuffle units and take the first num_units_to_sample
        keep_indices = np.random.permutation(num_units)[:num_units_to_sample]

        unit_mask = np.zeros_like(unit_ids, dtype=bool)
        unit_mask[keep_indices] = True

        data.units = data.units.select_by_mask(unit_mask)

        nested_attr = self.field.split(".")
        target_obj = getattr(data, nested_attr[0])
        if isinstance(target_obj, IrregularTimeSeries):
            # make a mask to select spikes that are from the units we want to keep
            spike_mask = np.isin(target_obj.unit_index, keep_indices)

            # using lazy masking, we will apply the mask for all attributes from spikes
            # and units.
            setattr(data, self.field, target_obj.select_by_mask(spike_mask))

            relabel_map = np.zeros(num_units, dtype=np.long)
            relabel_map[unit_mask] = np.arange(unit_mask.sum())

            target_obj = getattr(data, self.field)
            target_obj.unit_index = relabel_map[target_obj.unit_index]
        elif isinstance(target_obj, RegularTimeSeries):
            assert len(nested_attr) == 2
            setattr(target_obj, nested_attr[1], getattr(target_obj, nested_attr[1])[:, unit_mask])
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")

        return data
