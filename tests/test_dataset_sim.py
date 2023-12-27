import os
from collections import OrderedDict
from pathlib import Path

import msgpack
import numpy.testing as npt
import pytest
import torch
import torchtext
import util
from dateutil import parser
from torch.utils.data import DataLoader

from kirby.data import Dataset
from kirby.data.dataset import Collate
from kirby.models import PerceiverNM
from kirby.taxonomy import (
    ChunkDescription,
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    TrialDescription,
    decoder_registry,
    description_helper,
    to_serializable,
    weight_registry,
)
from kirby.taxonomy.taxonomy import Output, Task
from kirby.utils import move_to

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
        folds=["train", "val", "test"],
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
                        task=Task.CONTINUOUS_REACHING,
                        fields={Output.CURSOR2D: "cursor2d"},
                        trials=[
                            TrialDescription(
                                id="20100101_01_01",
                                chunks={
                                    "train": [
                                        ChunkDescription(
                                            id="20100101_01_01_01",
                                            duration=1,
                                            start_time=0,
                                        ),
                                        ChunkDescription(
                                            id="20100101_01_01_02",
                                            duration=1,
                                            start_time=1,
                                        ),
                                    ]
                                },
                                footprints={},
                            )
                        ],
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
                        task=Task.CONTINUOUS_REACHING,
                        fields={Output.FINGER3D: "finger3d"},
                        trials=[
                            TrialDescription(
                                id="20100102_01_01",
                                chunks={
                                    "train": [
                                        ChunkDescription(
                                            id="20100102_01_01_01",
                                            duration=1,
                                            start_time=0,
                                        ),
                                        ChunkDescription(
                                            id="20100102_01_01_02",
                                            duration=1,
                                            start_time=1,
                                        ),
                                        ChunkDescription(
                                            id="20100102_01_01_03",
                                            duration=1,
                                            start_time=1,
                                        ),
                                    ]
                                },
                                footprints={},
                            )
                        ],
                    )
                ],
                units=["e", "d"],
            ),
        ],
    )

    with open(tmp_path / id / "description.mpk", "wb") as f:
        msgpack.dump(
            to_serializable(struct), f, default=description_helper.encode_datetime
        )

    return tmp_path


def test_dataset_selection(description_mpk):
    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": {"dandiset": "odoherty_sabes"}}],
    )

    assert len(ds.chunk_info) == 5
    assert ds.chunk_info[0]["filename"] == (
        description_mpk / "odoherty_sabes" / "train" / "20100101_01_01_01.pt"
    )

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": {"dandiset": "odoherty_sabes", "subject": "alice"}}],
    )

    assert len(ds.chunk_info) == 2

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": {"dandiset": "odoherty_sabes", "sortset": "20100101"}}],
    )

    assert len(ds.chunk_info) == 2

    ds = Dataset(
        description_mpk,
        "train",
        [
            {
                "selection": {
                    "dandiset": "odoherty_sabes",
                    "session": "20100102_01",
                }
            }
        ],
    )

    assert len(ds.chunk_info) == 3

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": {"dandiset": "odoherty_sabes"}}],
    )

    assert len(ds.chunk_info) == 5

    ds = Dataset(
        description_mpk,
        "train",
        [{"selection": {"dandiset": "odoherty_sabes", "output": "CURSOR2D"}}],
    )

    assert len(ds.chunk_info) == 2