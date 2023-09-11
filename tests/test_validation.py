import os
from collections import OrderedDict
from pathlib import Path
import lightning

import torch
import torchtext

from kirby.data import Dataset
from kirby.data.dataset import Collate
from kirby.models import PerceiverNM
from kirby.taxonomy import (
    decoder_registry,
)
from kirby.utils import train_wrapper

DATA_ROOT = str(Path(os.environ["SLURM_TMPDIR"]) / "processed")

def test_validation():
    ds = Dataset(
        DATA_ROOT,
        "valid",
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
        behavior_type_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        sequence_length=1.0,
        unit_vocab=vocab,
        decoder_registry=decoder_registry,
    )
    
    model = PerceiverNM(
        max_num_units=len(vocab),
        session_names=ds.session_names,
        num_latents=16,
        task_specs=decoder_registry
    )
    model = model.to('cuda')

    wrapper = train_wrapper.TrainWrapper(
        model=model,
        optimizer=None,
        scheduler=None,
    )
    wrapper.to('cuda')

    trainer = lightning.Trainer(accelerator="cuda")

    validator = train_wrapper.CustomValidator(ds, collate_fn)
    validator.on_validation_epoch_start(trainer, wrapper)