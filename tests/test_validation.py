from pathlib import Path

import pytest
import lightning
import util
from torch.utils.data import DataLoader

from kirby.data import Dataset
from kirby.data.sampler import SequentialFixedWindowSampler
from kirby.data.collate import collate
from kirby.models import POYOPlus, POYOPlusTokenizer
from kirby.taxonomy import decoder_registry
from kirby.utils import train_wrapper

import util

DATA_ROOT = Path(util.get_data_paths()["processed_dir"]) / "processed"


# this test only passes when only one gpu is visible
def test_validation():
    model = POYOPlus(
        task_specs=decoder_registry,
        backend_config="cpu",
    )

    tokenizer = POYOPlusTokenizer(
        unit_tokenizer=model.unit_emb.tokenizer,
        session_tokenizer=model.session_emb.tokenizer,
        decoder_registry=decoder_registry,
        latent_step=1.0 / 8,
        num_latents_per_step=4,
        batch_type=["stacked", "stacked", "stacked"],
        eval=True,
    )

    ds = Dataset(
        DATA_ROOT,
        "valid",
        [
            {
                "selection": [
                    {
                        "dandiset": "mc_maze_small",
                        "session": "jenkins_20090928_maze",
                    }
                ],
                "config": {
                    "multitask_readout": [
                        {
                            "decoder_id": "ARMVELOCITY2D",
                            "weight": 2.0,
                            "subtask_key": None,
                            "metrics": [
                                {
                                    "metric": "r2",
                                    "task": "REACHING",
                                },
                            ]
                        }
                    ],
                },
            }
        ],
        transform=tokenizer,
    )

    model.unit_emb.initialize_vocab(ds.unit_ids)
    model.session_emb.initialize_vocab(ds.session_ids)

    sampler = SequentialFixedWindowSampler(
        interval_dict=ds.get_sampling_intervals(),
        window_length=1.0,
    )
    assert len(sampler) > 0

    loader = DataLoader(ds, collate_fn=collate, batch_size=16, sampler=sampler)

    device = "cpu"

    model = model.to(device)

    wrapper = train_wrapper.TrainWrapper(
        model=model,
        optimizer=None,
        scheduler=None,
    )
    wrapper.to(device)

    trainer = lightning.Trainer(accelerator=device)

    validator = train_wrapper.CustomValidator(loader)
    metrics = validator.on_validation_epoch_start(trainer, wrapper)

    assert "val_mc_maze_small/jenkins_20090928_maze_armvelocity2d_r2" == metrics.iloc[0].name


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
