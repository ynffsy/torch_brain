import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from model import (
    BhvrDecoder,
    ContextManager,
    Decoder,
    Encoder,
    MaeMaskManager,
    SpikesPatchifier,
    SslDecoder,
)
from omegaconf import OmegaConf, open_dict
from torch import optim
from torch.utils.data import DataLoader
from transforms import FilterUnit, Ndt2Tokenizer

from brainsets.taxonomy import decoder_registry
from temporaldata import Interval
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.transforms import Compose

log = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg,
        mae_mask_manager: Optional[MaeMaskManager],
        ctx_manager: ContextManager,
        spikes_patchifier: SpikesPatchifier,
        encoder: Encoder,
        decoder: Decoder,
    ):
        super().__init__()
        self.cfg = cfg
        self.mae_mask_manager = mae_mask_manager
        self.ctx_manager = ctx_manager
        self.spikes_patchifier = spikes_patchifier
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def training_step(self, batch, batch_idx):
        mae_loss = 0.0
        if "ssl" in batch:
            decoder_out = self._step(batch["ssl"], "ssl")
            self.log("train_shuffle_infill_loss", decoder_out["ssl_loss"])

        bhv_loss = 0.0
        if "superv" in batch:
            decoder_out = self._step(batch["superv"], "bhv")
            self.log("val_kinematic_decoding_loss", decoder_out["bhv_loss"])
            self.log("val_kinematic_r2_x", decoder_out["r2"][0])
            self.log("val_kinematic_r2_y", decoder_out["r2"][1])
            self.log("val_kinematic_r2", decoder_out["r2"].mean())

        loss = bhv_loss + mae_loss
        self.log("train_loss", loss, prog_bar=True)

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        mae_loss = 0.0
        if "ssl" in batch:
            decoder_out = self._step(batch["ssl"], "ssl")
            self.log("val_shuffle_infill_loss", decoder_out["ssl_loss"])

        bhv_loss = 0.0
        if "superv" in batch:
            decoder_out = self._step(batch["superv"], "bhv")
            self.log("val_kinematic_decoding_loss", decoder_out["bhv_loss"])
            self.log("val_kinematic_r2_x", decoder_out["r2"][0])
            self.log("val_kinematic_r2_y", decoder_out["r2"][1])
            self.log("val_kinematic_r2", decoder_out["r2"].mean())

        loss = mae_loss + bhv_loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def _step(self, batch, method: str = "ssl"):
        if method == "ssl":
            batch = self.mae_mask_manager(batch)
        encoder_input = self.spikes_patchifier(batch["spike_tokens"])
        ctx_emb = self.ctx_manager(batch, encoder_input.dtype)
        encoder_out = self.encoder(encoder_input, ctx_emb, batch)
        return self.decoder(encoder_out, ctx_emb, batch)

    def configure_optimizers(self):
        # TODO update to match the superv settings (*10 lr for decoder layer and no scheduler)
        cfg = self.cfg.optimizer

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        linearLR = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.start_factor, total_iters=cfg.warmup_steps
        )
        cosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.decay_steps, eta_min=cfg.lr_min
        )
        scheduler = optim.lr_scheduler.ChainedScheduler([linearLR, cosineAnnealingLR])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    # params = list(self.named_parameters())


# # As of 2/24/23 all my parameters are named, this better stay the case
# accel_flag = lambda name: name in self.novel_params or (
#     "session_embed" in name
#     or "subject_embed" in name
#     or "task_embed" in name
#     or "array_embed" in name
# )
# grouped_params = [
#     {
#         "params": [p for n, p in params if accel_flag(n)],
#         "lr": self.cfg.lr_init * self.cfg.accelerate_new_params,
#     },
#     {
#         "params": [p for n, p in params if not accel_flag(n)],
#         "lr": self.cfg.lr_init,
#     },
# ]


class DataModule(L.LightningDataModule):
    def __init__(
        self, cfg, tokenizer: Ndt2Tokenizer, is_ssl: bool = True, unsorted: bool = True
    ):
        super().__init__()
        self.cfg = cfg
        self.is_ssl = is_ssl
        if is_ssl:
            self.dataset_cfg = cfg.data_ssl
        else:
            self.dataset_cfg = cfg.data_superv

        if not unsorted:
            raise NotImplementedError("Only unsorted data is supported")

        self.unsorted = unsorted
        keep_M1_unit = FilterUnit("/M1", keep=True)
        self.transforms = Compose([keep_M1_unit, tokenizer])

    def setup(self, stage=None):
        cfg = self.cfg

        # do not use split for dataset because is handle at sampler level
        self.dataset = Dataset(
            root=cfg.data_root,
            split=None,
            config=self.dataset_cfg,
            transform=self.transforms,
        )
        self.dataset.disable_data_leakage_check()

        self.train_intervals: Dict[str, List[Tuple[float, float]]]
        self.eval_intervals: Dict[str, List[Tuple[float, float]]]
        intervals = self.ndt2_custom_sampling_intervals()
        self.train_intervals, self.eval_intervals = intervals

    def get_ctx_vocab(self, ctx_keys):
        return {k: getattr(self.dataset, f"get_{k}_ids")() for k in ctx_keys}

    def train_dataloader(self):
        cfg = self.cfg
        train_sampler = RandomFixedWindowSampler(
            interval_dict=self.train_intervals,
            window_length=cfg.ctx_time,
            generator=torch.Generator(),
        )
        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        return train_loader

    def val_dataloader(self):
        cfg = self.cfg
        val_sampler = SequentialFixedWindowSampler(
            interval_dict=self.eval_intervals,
            window_length=cfg.ctx_time,
            drop_short=True,
        )

        bs = cfg.batch_size_per_gpu if self.is_ssl else cfg.superv_batch_size_per_gpu
        val_loader = DataLoader(
            dataset=self.dataset,
            batch_size=bs,
            sampler=val_sampler,
            collate_fn=collate,
            num_workers=cfg.num_workers,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        return val_loader

    def test_dataloader(self):
        return None

    def ndt2_custom_sampling_intervals(
        self,
    ) -> Tuple[Dict, Dict]:
        """
        Custom sampling intervals for NDT2.
        It splits the dataset into training and validation sets.
        Note: Used at the sampling level and not at the session level.
        This is because ndt2 split at the dataset object level and not at session level.
        """
        ses_keys = []
        dataset = self.dataset
        ctx_time = self.cfg.ctx_time
        train_ratio = self.cfg.train_ratio
        seed = self.cfg.split_seed

        for ses_id, ses in dataset._data_objects.items():
            nb_trials = int(ses.domain.end[-1] - ses.domain.start[0])
            for i in range(nb_trials):
                ses_keys.append(f"{ses_id}-{i}")

        L.seed_everything(seed)
        np.random.shuffle(ses_keys)
        tv_cut = int(train_ratio * len(ses_keys))
        train_keys, val_keys = ses_keys[:tv_cut], ses_keys[tv_cut:]

        def get_dict(keys):
            d = defaultdict(list)
            for k in keys:
                ses_id, trial = k.split("-")
                ses = dataset._data_objects[ses_id]
                ses_start = ses.domain.start[0]
                offset = ctx_time * int(trial)
                start = ses_start + offset
                end = start + ctx_time
                d[ses_id].append((start, end))
            return dict(d)

        train_sampling_intervals = get_dict(train_keys)
        val_sampling_intervals = get_dict(val_keys)

        # val will be deterministic and need to be sorted
        for v in val_sampling_intervals.values():
            v.sort()
        val_sampling_intervals = dict(sorted(val_sampling_intervals.items()))

        # TODO this is very dirty code should be cleaned
        def list_to_inter(l):
            start = np.array([e[0] for e in l])
            end = np.array([e[1] for e in l])
            return Interval(start, end)

        def to_inter(d):
            return {k: list_to_inter(v) for k, v in d.items()}

        train_sampling_intervals = to_inter(train_sampling_intervals)
        val_sampling_intervals = to_inter(val_sampling_intervals)

        return train_sampling_intervals, val_sampling_intervals


def set_wandb(cfg, log) -> Optional[WandbLogger]:
    if not cfg.wandb.enable:
        return None
    wandb_logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        save_dir=cfg.log_dir,
        log_model=False,
    )
    log.info(f"Using wandb logger: {wandb_logger.version}")

    return wandb_logger


def set_callbacks(cfg) -> List[Callback]:
    cfg = cfg.checkpoint
    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.enable:
        callbacks.append(
            ModelCheckpoint(
                save_last=True,  # saves a checkpoint for the last epoch
                every_n_train_steps=cfg.every_n_steps,
                every_n_epochs=cfg.every_n_epochs,
                save_top_k=-1 if cfg.save_all else 1,
            )
        )
    return callbacks


def run_training(cfg):
    L.seed_everything(cfg.seed)

    if cfg.fast_dev_run:
        cfg.wandb.enable = False
        cfg.num_workers = 0

    with open_dict(cfg):
        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        log.info(f"Number of GPUs: {num_gpus}")
        log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    wandb_logger = set_wandb(cfg, log)
    dim = cfg.model.dim

    # Mask manager (for MAE SSL)
    mae_mask_manager = None
    if cfg.is_ssl:
        mae_mask_manager = MaeMaskManager(cfg.mask_ratio)

    # context manager
    ctx_manager = ContextManager(dim)

    # Spikes patchifier
    spikes_patchifier = SpikesPatchifier(dim, cfg.patch_size)

    # Model = Encoder + Decoder
    encoder = Encoder(
        dim=dim,
        max_time_patches=cfg.model.max_time_patches,
        max_space_patches=cfg.model.max_space_patches,
        **cfg.model.encoder,
    )

    if cfg.is_ssl:
        decoder = SslDecoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            patch_size=cfg.patch_size,
            **cfg.model.predictor,
        )
    else:
        decoder = BhvrDecoder(
            dim=dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            bin_time=cfg.bin_time,
            **cfg.model.bhv_decoder,
        )

    # Train wrapper
    train_wrapper = TrainWrapper(
        cfg, mae_mask_manager, ctx_manager, spikes_patchifier, encoder, decoder
    )

    # Tokenizer
    ctx_tokenizer = ctx_manager.get_ctx_tokenizer()
    tokenizer = Ndt2Tokenizer(
        ctx_time=cfg.ctx_time,
        bin_time=cfg.bin_time,
        patch_size=cfg.patch_size,
        pad_val=cfg.pad_val,
        decoder_registry=decoder_registry,
        mask_ratio=cfg.mask_ratio,
        ctx_tokenizer=ctx_tokenizer,
        inc_behavior=not cfg.is_ssl,
        inc_mask=cfg.is_ssl,
    )

    # set up data module
    data_module = DataModule(cfg, tokenizer, cfg.is_ssl)
    data_module.setup()

    # register context
    ctx_manager.init_vocab(data_module.get_ctx_vocab(ctx_manager.keys))

    L.seed_everything(cfg.seed)

    # Callbacks
    callbacks = set_callbacks(cfg)

    # Set up trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        accelerator="gpu",
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        strategy="ddp_find_unused_parameters_true",
    )

    if wandb_logger:
        wandb_logger.watch(train_wrapper, log="all")

    # Train model
    trainer.fit(train_wrapper, data_module)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
