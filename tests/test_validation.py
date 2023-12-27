import os
from collections import OrderedDict
from pathlib import Path

import lightning
import torch
import torchtext
from omegaconf import DictConfig, OmegaConf

from kirby.data import Dataset
from kirby.data.dataset import Collate
from kirby.models import PerceiverNM
from kirby.taxonomy import decoder_registry, weight_registry
from kirby.utils import train_wrapper

import util

DATA_ROOT = Path(util.get_data_paths()["processed_dir"]) / "processed"


# this test only passes when only one gpu is visible
def test_validation():
    ds = Dataset(
        DATA_ROOT,
        "valid",
        OmegaConf.create(
            [
                {
                    "selection": {
                        "dandiset": "mc_maze_small",
                        "session": "jenkins_20090928_maze",
                    },
                    "metrics": [{"output_key": "ARMVELOCITY2D"}],
                }
            ]
        ),
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

    model = PerceiverNM(
        unit_vocab=vocab,
        session_names=ds.session_names,
        num_latents=16,
        task_specs=decoder_registry,
    )
    model = model.to("cuda")

    wrapper = train_wrapper.TrainWrapper(
        model=model,
        optimizer=None,
        scheduler=None,
    )
    wrapper.to("cuda")

    trainer = lightning.Trainer(accelerator="cuda")

    validator = train_wrapper.CustomValidator(ds, collate_fn)
    r2 = validator.on_validation_epoch_start(trainer, wrapper)

    assert "mse_armvelocity2d" in r2.columns

    ds = Dataset(
        DATA_ROOT,
        "valid",
        OmegaConf.create(
            [
                {
                    "selection": {
                        "dandiset": "mc_maze_small",
                        "session": "jenkins_20090928_maze",
                    },
                    "metrics": [{"output_key": "ARMVELOCITY2D", "metric": "r2"}],
                }
            ]
        ),
    )

    validator = train_wrapper.CustomValidator(ds, collate_fn)
    r2 = validator.on_validation_epoch_start(trainer, wrapper)

    assert "r2_armvelocity2d" in r2.columns


# def test_validation_willett():
#     ds = Dataset(
#         DATA_ROOT,
#         "valid",
#         OmegaConf.create([
#             {
#                 "selection": {
#                     "dandiset": "willett_shenoy",
#                     "session": "willett_shenoy_t5/t5.2019.05.08_single_letters",
#                 },
#                 "metrics": [{"output_key": "WRITING_CHARACTER"}],
#             }
#         ]),
#     )
#     batch_size = 16

#     od = OrderedDict({x: 1 for x in ds.unit_names})
#     vocab = torchtext.vocab.vocab(od, specials=["NA"])

#     collate_fn = Collate(
#         num_latents_per_step=16,  # This was tied in train_poyo_1.py
#         step=1.0 / 8,
#         sequence_length=1.0,
#         unit_vocab=vocab,
#         decoder_registry=decoder_registry,
#         weight_registry=weight_registry,
#     )

#     model = PerceiverNM(
#         unit_vocab=vocab,
#         session_names=ds.session_names,
#         num_latents=16,
#         task_specs=decoder_registry
#     )
#     model = model.to('cuda')

#     wrapper = train_wrapper.TrainWrapper(
#         model=model,
#         optimizer=None,
#         scheduler=None,
#     )
#     wrapper.to('cuda')

#     trainer = lightning.Trainer(accelerator="cuda")

#     validator = train_wrapper.CustomValidator(ds, collate_fn)
#     r2 = validator.on_validation_epoch_start(trainer, wrapper)
#     assert "bce_writing_character" in r2.columns
