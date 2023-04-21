
import torch
from einops import repeat

class UnitDropout:
    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1, f'p must be between 0 and 1, got {p}'
        self.p = p

    def __call__(self, data, num_units=None):
        spikes = data.spikes
        target_rates = data.trials.target_rates

        if num_units is None:
            maybe_num_units = data.spikes.max() + 1
            num_units = maybe_num_units

        # sample units to drop
        unit_mask = torch.rand(num_units) > self.p

        # drop units from spikes
        spike_mask = unit_mask[spikes.unit_id]
        out = spikes.copy()
        out.timestamps = spikes.timestamps[spike_mask]
        out.unit_id = spikes.unit_id[spike_mask]
        data.spikes = out

        # make target rate for dropped units
        t = target_rates.size(0)
        target_rates = target_rates[:, ~unit_mask].reshape(-1)
        target_units = torch.arange(num_units)[~unit_mask]
        target_units = repeat(target_units, 'u -> t u', t=t)
        data.target_rates = target_rates
        data.target_units = target_units
        return data
