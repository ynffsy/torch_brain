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
    names = ["a"] * 10 + ["b"] * 20 + ["c"] * 70
    names = np.array(names)
    types = torch.zeros(100)

    for i in range(100):
        data = Data(
            spikes=IrregularTimeSeries(
                timestamps=timestamps,
                names=names,
                types=types,
            ),
            units=Data(
                unit_name=["a", "b", "c"],
            ),
        )
        do = UnitDropout(min_units=1, mode_units=2, max_units=3)
        data_t = do(data)
        assert data_t.spikes.timestamps.shape[0] in (10, 20, 30, 70, 80, 90, 100)
        print(np.unique(data_t.spikes.names), data_t.units.unit_name)
        assert len(np.unique(data_t.spikes.names)) == len(data_t.units.unit_name)
        assert len(data_t.spikes.timestamps) == len(data_t.spikes.names)


def test_spikes_w_precomputed_index_maps():
    timestamps = torch.zeros(100)
    names = ["a"] * 10 + ["b"] * 20 + ["c"] * 70
    names = np.array(names)
    types = torch.zeros(100)

    for i in range(100):
        data = Data(
            spikes=IrregularTimeSeries(
                timestamps=timestamps,
                names=names,
                types=types,
            ),
            units=Data(
                unit_name=["a", "b", "c"],
            ),
        )

        data.spikes.precompute_index_map('names')

        do = UnitDropout(min_units=1, mode_units=2, max_units=3)
        data_t = do(data)
        assert data_t.spikes.timestamps.shape[0] in (10, 20, 30, 70, 80, 90, 100)
        print(np.unique(data_t.spikes.names), data_t.units.unit_name)
        assert len(np.unique(data_t.spikes.names)) == len(data_t.units.unit_name)
        assert len(data_t.spikes.timestamps) == len(data_t.spikes.names)
