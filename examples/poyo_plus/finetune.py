import logging
from typing import List, Optional

import numpy as np
import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig

from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
# from torch_brain.utils.datamodules import DataModule
# from torch_brain.utils.stitcher import StitchEvaluator
from torch_brain.utils.stitcher import (
    MultiTaskDecodingStitchEvaluator,
    DataForMultiTaskDecodingStitchEvaluator,
)

from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch.utils.data import DataLoader
from torch_brain.data import Dataset, collate
from torch_brain.transforms import Compose
from train import TrainWrapper, DataModule

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


class GradualUnfreezing(L.Callback):
    r"""A Lightning callback to handle freezing and unfreezing of the model for the
    purpose of finetuning the model to new sessions. If this callback is used,
    most of the model weights will be frozen initially.
    The only parts of the model that will be left unforzen are the unit, and session embeddings.
    One we reach the specified epoch (`unfreeze_at_epoch`), the entire model will be unfrozen.
    """

    _has_been_frozen: bool = False
    frozen_params: Optional[List[nn.Parameter]] = None

    def __init__(self, unfreeze_at_epoch: int):
        self.enabled = unfreeze_at_epoch != 0
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.cli_log = logging.getLogger(__name__)

    @classmethod
    def freeze(cls, model):
        r"""Freeze the model weights, except for the unit and session embeddings, and
        return the list of frozen parameters.
        """
        layers_to_freeze = [
            model.enc_atn,
            model.enc_ffn,
            model.proc_layers,
            model.dec_atn,
            model.dec_ffn,
            model.readout,
            model.token_type_emb,
            model.task_emb,
        ]

        frozen_params = []
        for layer in layers_to_freeze:
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params.append(param)

        return frozen_params

    def on_train_start(self, trainer, pl_module):
        if self.enabled:
            self.frozen_params = self.freeze(pl_module.model)
            self._has_been_frozen = True
            self.cli_log.info(
                f"POYO+ Perceiver frozen at epoch 0. "
                f"Will stay frozen until epoch {self.unfreeze_at_epoch}."
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled and (trainer.current_epoch == self.unfreeze_at_epoch):
            if not self._has_been_frozen:
                raise RuntimeError("Model has not been frozen yet.")

            for param in self.frozen_params:
                param.requires_grad = True

            self.frozen_params = None
            self.cli_log.info(
                f"POYO+ Perceiver unfrozen at epoch {trainer.current_epoch}"
            )


class FinetuneDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, model):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)
        
        # You might or might not need these
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sequence_length = None
        self.model = model

    def prepare_data(self):
        # Typically handle any downloads, etc. here. Possibly empty if not needed.
        pass

    def setup(self, stage=None):
        self.sequence_length = self.model.sequence_length

        # prepare transforms
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        # compose transforms, tokenizer is always the last transform
        train_transform = Compose([*train_transforms, self.model.tokenize])
        eval_transform = Compose([*eval_transforms, self.model.tokenize])

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=train_transform,
        )
        self.train_dataset.disable_data_leakage_check()

        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.eval_dataset,
            split="valid",
            transform=eval_transform,
        )
        self.val_dataset.disable_data_leakage_check()

        # create the test dataset
        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.eval_dataset,
            split="test",
            transform=eval_transform,
        )
        self.test_dataset.disable_data_leakage_check()


    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        return train_loader


    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        self.val_sequence_index = val_sampler.sequence_index

        return val_loader
    

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader

    def get_unit_ids(self):
        # return self.test_dataset.get_unit_ids()
        return self.train_dataset.get_unit_ids()

    def get_session_ids(self):
        # return self.test_dataset.get_session_ids()
        return self.train_dataset.get_session_ids()

    def get_recording_config_dict(self):
        return self.test_dataset.get_recording_config_dict()
        # return self.train_dataset.get_recording_config_dict()
    
    # Optionally, if you need the same metrics approach:
    def get_metrics(self):
        from collections import defaultdict
        dataset_config_dict = self.get_recording_config_dict()
        metrics = defaultdict(lambda: defaultdict(dict))
        for recording_id, recording_config in dataset_config_dict.items():
            for readout_config in recording_config["multitask_readout"]:
                readout_id = readout_config["readout_id"]
                for metric_config in readout_config["metrics"]:
                    metric = hydra.utils.instantiate(metric_config["metric"])
                    metrics[recording_id][readout_id][str(metric)] = metric
        return metrics


def load_model_from_ckpt(model: nn.Module, ckpt_path: str) -> None:
    if ckpt_path is None:
        raise ValueError("Must provide a checkpoint path to finetune the model.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)


@hydra.main(version_base="1.3", config_path="./configs", config_name="finetune_poyo_mp.yaml")
def main(cfg: DictConfig):
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype("str")),
    ])

    # make model
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)
    load_model_from_ckpt(model, cfg.ckpt_path)
    log.info(f"Loaded model weights from {cfg.ckpt_path}")

    # setup data module
    # data_module = DataModule(cfg)
    # data_module.setup_dataset_and_link_model(model)

    data_module = FinetuneDataModule(cfg, model)
    data_module.setup()

    # register units and sessions
    # gather all unit IDs
    train_unit_ids = data_module.train_dataset.get_unit_ids()
    val_unit_ids   = data_module.val_dataset.get_unit_ids()
    test_unit_ids  = data_module.test_dataset.get_unit_ids()

    all_unit_ids = list(set(train_unit_ids + val_unit_ids + test_unit_ids))
    model.unit_emb.extend_vocab(all_unit_ids, exist_ok=True)
    model.unit_emb.subset_vocab(all_unit_ids)

    # do the same for session IDs
    train_session_ids = data_module.train_dataset.get_session_ids()
    val_session_ids   = data_module.val_dataset.get_session_ids()
    test_session_ids  = data_module.test_dataset.get_session_ids()

    all_session_ids = list(set(train_session_ids + val_session_ids + test_session_ids))
    model.session_emb.extend_vocab(all_session_ids, exist_ok=True)
    model.session_emb.subset_vocab(all_session_ids)


    # Lightning train wrapper
    wrapper = TrainWrapper(cfg=cfg, model=model)

    evaluator = MultiTaskDecodingStitchEvaluator(metrics=data_module.get_metrics())

    callbacks = [
        evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
        GradualUnfreezing(cfg.freeze_perceiver_until_epoch),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        num_sanity_val_steps=0,
        limit_val_batches=None,  # Ensure no limit on validation batches
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train
    trainer.fit(wrapper, data_module)

    # Test
    trainer.test(wrapper, data_module, "best")


if __name__ == "__main__":
    main()
