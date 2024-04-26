from pathlib import Path

import msgpack
import pytest
import util
from dateutil import parser
import numpy as np
import h5py

from kirby.data import Dataset, Data, IrregularTimeSeries, Interval, RegularTimeSeries
from kirby.data.dataset_builder import encode_datetime
from kirby.taxonomy import (
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    to_serializable,
)
from kirby.taxonomy import Task


DATA_ROOT = Path(util.get_data_paths()["uncompressed_dir"]) / "uncompressed"


@pytest.fixture
def description_mpk(tmp_path):
    id = "odoherty_sabes"
    (tmp_path / id).mkdir()
    struct = DandisetDescription(
        id=id,
        origin_version="0.0.0",
        derived_version="0.0.0",
        metadata_version="0.0.0",
        source="https://dandiarchive.org/#/dandiset/000005",
        description="",
        splits=["train", "val", "test"],
        subjects=[],
        sortsets=[
            SortsetDescription(
                id="20100101",
                subject="alice",
                areas=[],
                recording_tech=[],
                sessions=[
                    SessionDescription(
                        id="20100101_01",
                        recording_date=parser.parse("2010-01-01T00:00:00"),
                        task=Task.REACHING,
                        splits={"train": [(0, 1), (2, 3)], "full": [(0, 5)]},
                        trials=[],
                    )
                ],
                units=["a", "b", "c"],
            ),
            SortsetDescription(
                id="20100102",
                subject="bob",
                areas=[],
                recording_tech=[],
                sessions=[
                    SessionDescription(
                        id="20100102_01",
                        recording_date=parser.parse("2010-01-01T00:00:00"),
                        task=Task.REACHING,
                        splits={"train": [(0, 1), (2, 3), (3, 4)], "full": [(0, 5)]},
                        trials=[],
                    )
                ],
                units=["e", "d"],
            ),
        ],
    )

    with open(tmp_path / id / "description.mpk", "wb") as f:
        msgpack.dump(
            to_serializable(struct),
            f,
            default=encode_datetime,
        )

    # Create dummy session files
    for sortset in struct.sortsets:
        for session in sortset.sessions:
            filename = tmp_path / id / f"{session.id}.h5"
            dummy_data = Data(
                spikes=IrregularTimeSeries(
                    timestamps=np.arange(0, 5, 0.1),
                    domain="auto",
                ),
                trials=Interval(
                    start=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                    end=np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                    some_values=np.arange(5),
                ),
                traces=RegularTimeSeries(
                    values=np.arange(100),
                    sampling_rate=20,
                    domain=Interval(0, 5.0),
                ),
                domain=Interval(0, 5.0),
            )
            dummy_data.add_split_mask("train", Interval.from_list([(0, 1), (2, 3)]))
            with h5py.File(filename, "w") as f:
                dummy_data.to_hdf5(f)

    return tmp_path


def test_dataset_selection(description_mpk):
    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": [{"dandiset": "odoherty_sabes"}]}],
    )
    assert len(ds.session_info_dict) == 2

    assert ds.session_info_dict["odoherty_sabes/20100101_01"]["filename"] == (
        description_mpk / "odoherty_sabes" / "20100101_01.h5"
    )
    assert (
        len(ds.session_info_dict["odoherty_sabes/20100101_01"]["sampling_intervals"])
        == 2
    )

    assert ds.session_info_dict["odoherty_sabes/20100102_01"]["filename"] == (
        description_mpk / "odoherty_sabes" / "20100102_01.h5"
    )
    assert (
        len(ds.session_info_dict["odoherty_sabes/20100102_01"]["sampling_intervals"])
        == 3
    )

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": [{"dandiset": "odoherty_sabes", "subject": "alice"}]}],
    )
    assert len(ds.session_info_dict) == 1
    assert ds.session_info_dict["odoherty_sabes/20100101_01"]["filename"] == (
        description_mpk / "odoherty_sabes" / "20100101_01.h5"
    )
    assert (
        len(ds.session_info_dict["odoherty_sabes/20100101_01"]["sampling_intervals"])
        == 2
    )

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": [{"dandiset": "odoherty_sabes", "sortset": "20100101"}]}],
    )
    assert len(ds.session_info_dict) == 1
    assert ds.session_info_dict["odoherty_sabes/20100101_01"]["filename"] == (
        description_mpk / "odoherty_sabes" / "20100101_01.h5"
    )
    assert (
        len(ds.session_info_dict["odoherty_sabes/20100101_01"]["sampling_intervals"])
        == 2
    )

    ds = Dataset(
        description_mpk,
        "train",
        [
            {
                "selection": [
                    {
                        "dandiset": "odoherty_sabes",
                        "session": "20100102_01",
                    }
                ]
            }
        ],
    )

    assert ds.session_info_dict["odoherty_sabes/20100102_01"]["filename"] == (
        description_mpk / "odoherty_sabes" / "20100102_01.h5"
    )
    assert (
        len(ds.session_info_dict["odoherty_sabes/20100102_01"]["sampling_intervals"])
        == 3
    )


def test_get_session_data(description_mpk):
    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": [{"dandiset": "odoherty_sabes"}]}],
    )

    data = ds.get_session_data("odoherty_sabes/20100101_01")

    assert len(data.spikes) == 20
    assert np.allclose(
        data.spikes.timestamps,
        np.concatenate([np.arange(0, 1, 0.1), np.arange(2, 3, 0.1)]),
    )

    assert len(data.trials) == 2
    assert np.allclose(data.trials.start, [0.0, 2.0])
    assert np.allclose(data.trials.end, [0.5, 2.5])
    assert np.allclose(data.trials.some_values, [0, 2])

    assert len(data.traces) == 40
    assert isinstance(data.traces, IrregularTimeSeries)
    assert np.allclose(
        data.traces.timestamps,
        np.concatenate([np.arange(0, 1, 0.05), np.arange(2, 3, 0.05)]),
    )
    assert np.allclose(
        data.traces.values,
        np.concatenate([np.arange(20), np.arange(40, 60)]),
    )

    ds = Dataset(
        description_mpk,
        "full",
        [{"selection": [{"dandiset": "odoherty_sabes"}]}],
    )

    data = ds.get_session_data("odoherty_sabes/20100101_01")

    assert len(data.spikes) == 50
    assert np.allclose(data.spikes.timestamps, np.arange(0, 5, 0.1))

    assert len(data.trials) == 5
    assert np.allclose(data.trials.start, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(data.trials.end, np.array([0.5, 1.5, 2.5, 3.5, 4.5]))
    assert np.allclose(data.trials.some_values, np.arange(5))

    assert isinstance(data.traces, RegularTimeSeries)
    assert len(data.traces) == 100
    assert np.allclose(data.traces.values, np.arange(100))
