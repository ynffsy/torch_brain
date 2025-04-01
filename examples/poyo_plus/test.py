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

import ipdb



# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)



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


class TestDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, model):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)
        
        # You might or might not need these
        self.test_dataset = None
        self.sequence_length = None
        self.model = model

    def prepare_data(self):
        # Typically handle any downloads, etc. here. Possibly empty if not needed.
        pass

    def setup(self, stage=None):
        self.sequence_length = self.model.sequence_length

        # prepare transforms
        transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        # compose transforms, tokenizer is always the last transform
        test_transform = Compose([*transforms, self.model.tokenize])

        # 3) Create the test dataset
        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            # split="test",  # or however you designate test set
            transform=test_transform,
        )
        self.test_dataset.disable_data_leakage_check()
            

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
        return self.test_dataset.get_unit_ids()

    def get_session_ids(self):
        return self.test_dataset.get_session_ids()

    def get_recording_config_dict(self):
        return self.test_dataset.get_recording_config_dict()

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



@hydra.main(version_base="1.3", config_path="./configs", config_name="test_poyo_mp.yaml")
def main(cfg: DictConfig):
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype("str")),
    ])

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

    # make model
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)
    load_model_from_ckpt(model, cfg.ckpt_path)
    log.info(f"Loaded model weights from {cfg.ckpt_path}")

    # 3) Build the TestDataModule
    #    Notice that we pass `model.tokenize` so we can transform test data consistently
    test_data_module = TestDataModule(cfg, model)
    test_data_module.setup()

    # 4) Update model’s session/unit vocab from test data
    unit_ids = test_data_module.get_unit_ids()
    session_ids = test_data_module.get_session_ids()
    model.unit_emb.extend_vocab(unit_ids, exist_ok=True)
    model.unit_emb.subset_vocab(unit_ids)
    model.session_emb.extend_vocab(session_ids, exist_ok=True)
    model.session_emb.subset_vocab(session_ids)

    # 5) Build your TrainWrapper or whatever LightningModule you want to test
    wrapper = TrainWrapper(cfg=cfg, model=model)

    # 6) Build your evaluation callback (for metrics, etc.)
    evaluator = MultiTaskDecodingStitchEvaluator(
        metrics=test_data_module.get_metrics()
    )

    # 7) Build trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        callbacks=[evaluator],  # keep it minimal
        # any other trainer configs
    )

    # 8) Run test
    trainer.test(wrapper, datamodule=test_data_module)



if __name__ == "__main__":
    main()
