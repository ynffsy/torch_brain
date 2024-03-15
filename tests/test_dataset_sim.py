from pathlib import Path

import msgpack
import pytest
import util
from dateutil import parser
import numpy as np
import h5py

from kirby.data import Dataset, Data, IrregularTimeSeries, Interval
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
                        splits={"train": [(0, 1), (1, 2)]},
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
                        splits={"train": [(0, 1), (1, 2), (2, 3)]},
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
                    timestamps=np.arange(0, 1, 0.1),
                    domain="auto",
                ),
                domain=Interval(0, 1),
            )
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
