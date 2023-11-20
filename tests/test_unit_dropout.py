import numpy as np
import torch
from kirby.data.data import Data, IrregularTimeSeries
from kirby.transforms.unit_dropout import UnitCustomDistribution, UnitDropout


def test_distro():
    for i in range(100):
        num_units = UnitCustomDistribution(
            min_units=100, mode_units=150, max_units=200
        ).sample(196)
        assert num_units >= 100 and num_units <= 200


def test_spikes():
    timestamps = torch.zeros(100)
    unit_index = [0] * 10 + [1] * 20 + [2] * 70
    unit_index = torch.tensor(unit_index)
    types = torch.zeros(100)

    for i in range(100):
        data = Data(
            spikes=IrregularTimeSeries(
                timestamps=timestamps,
                unit_index=unit_index,
                types=types,
            ),
            units=Data(
                unit_name=["a", "b", "c"],
            ),
        )
        do = UnitDropout(min_units=1, mode_units=2, max_units=3)
        data_t = do(data)
        assert data_t.spikes.timestamps.shape[0] in (10, 20, 30, 70, 80, 90, 100)
        print(np.unique(data_t.spikes.unit_index), data_t.units.unit_name)
        assert len(data_t.units.unit_name) == 3  # We don't currently remove units
        assert len(data_t.spikes.timestamps) == len(data_t.spikes.unit_index)

        for _, value in data.spikes.__dict__.items():
            if value is None:
                continue
            
            assert value.shape[0] == len(data_t.spikes.timestamps)
