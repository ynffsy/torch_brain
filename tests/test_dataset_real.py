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

def test_load_real_data():
    ds = Dataset(DATA_ROOT, "train", [{"selection": {"dandiset": "perich_miller"}}])
    assert ds[0].start >= 0


def test_collate_data():
    ds = Dataset(
        DATA_ROOT,
        "train",
        [
            {
                "selection": {
                    "dandiset": "odoherty_sabes",
                    "session": "odoherty_sabes_reaching_2017_indy_20160921_01",
                },
                "metrics": [{"output_key": "CURSOR2D", "weight": 2.0}],
            }
        ],
    )
    assert len(ds) > 0

    od = OrderedDict({x: 1 for x in ds.unit_names})
    vocab = torchtext.vocab.vocab(od, specials=["NA"])

    collate_fn = Collate(
        num_latents_per_step=128,  # This was tied in train_poyo_1.py
        step=1.0 / 8,
        sequence_length=128,
        unit_vocab=vocab,
        decoder_registry=decoder_registry,
        weight_registry=weight_registry,
    )
    train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=16)
    for data in train_loader:
        assert data["spike_timestamps"].shape[0] == 16
        npt.assert_allclose(
            data["output_weights"]["CURSOR2D"].detach().cpu().numpy(), 2.0
        )
        break


def test_collate_data_willett():
    print("test_collate_data_willett")
    ds = Dataset(
        DATA_ROOT,
        "train",
        [
            {
                "selection": {
                    "dandiset": "willett_shenoy",
                    "sortset": "willett_shenoy_t5/t5.2019.05.08",
                },
                "metrics": [{"output_key": "WRITING_CHARACTER"}],
            }
        ],
    )
    assert len(ds) > 0

    od = OrderedDict({x: 1 for x in ds.unit_names})
    vocab = torchtext.vocab.vocab(od, specials=["NA"])

    collate_fn = Collate(
        num_latents_per_step=128,  # This was tied in train_poyo_1.py
        step=1.0 / 8,
        sequence_length=128,
        unit_vocab=vocab,
        decoder_registry=decoder_registry,
        weight_registry=weight_registry,
    )
    train_loader = DataLoader(
        ds, collate_fn=collate_fn, batch_size=4, drop_last=True, shuffle=True
    )
    for i, data in enumerate(train_loader):
        print(i)
        assert data["spike_timestamps"].shape[0] == 4
        npt.assert_allclose(
            data["output_weights"]["WRITING_CHARACTER"].detach().cpu().numpy(),
            1.0,
        )


def test_collate_data_perich():
    ds = Dataset(
        DATA_ROOT,
        "train",
        [
            {
                "selection": {
                    "dandiset": "perich_miller",
                    "sortset": "chewie_20161013",
                },
                "metrics": [{"output_key": "CURSOR2D"}],
            }
        ],
    )
    assert len(ds) > 0

    od = OrderedDict({x: 1 for x in ds.unit_names})
    vocab = torchtext.vocab.vocab(od, specials=["NA"])

    collate_fn = Collate(
        num_latents_per_step=128,  # This was tied in train_poyo_1.py
        step=1.0 / 8,
        sequence_length=128,
        unit_vocab=vocab,
        decoder_registry=decoder_registry,
        weight_registry=weight_registry,
    )
    train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=16)
    for data in train_loader:
        assert data["spike_timestamps"].shape[0] == 16
        npt.assert_allclose(
            data["output_weights"]["CURSOR2D"].detach().cpu().numpy().max(),
            50.0,
        )
        npt.assert_allclose(
            data["output_weights"]["CURSOR2D"].detach().cpu().numpy().min(),
            1.0,
        )
        break


def test_collated_data_model():
    ds = Dataset(
        DATA_ROOT,
        "train",
        [
            {
                "selection": {
                    "dandiset": "odoherty_sabes",
                    "session": "odoherty_sabes_reaching_2017_indy_20160921_01",
                },
                "metrics": [{"output_key": "CURSOR2D"}],
            }
        ],
    )
    batch_size = 16

    od = OrderedDict({x: 1 for x in ds.unit_names})
    vocab = torchtext.vocab.vocab(od, specials=["NA"])

    collate_fn = Collate(
        num_latents_per_step=16,  # This was tied in train_poyo_1.py
        step=1.0 / 8,
        sequence_length=1.0,
        unit_vocab=vocab,
        decoder_registry=decoder_registry,
        weight_registry=weight_registry,
    )
    train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)
    model = PerceiverNM(
        unit_vocab=vocab,
        session_names=ds.session_names,
        num_latents=16,
        task_specs=decoder_registry,
    )
    model = model.to("cuda")
    assert len(train_loader) > 0
    for data in train_loader:
        data = move_to(data, "cuda")
        output, loss, losses_taskwise = model(**data, compute_loss=True)

        for taskname in data["output_values"].keys():
            assert data["output_values"][taskname].shape[0] == output[taskname].shape[0]
        assert loss.shape == torch.Size([])  # it should be a scalar
        assert len(losses_taskwise.keys()) == len(data["output_values"].keys())
        break
