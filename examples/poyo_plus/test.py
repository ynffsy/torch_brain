import logging
from typing import List, Optional

import numpy as np
import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    Callback,
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

from train import TrainWrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            config=self.cfg.eval_dataset,
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



class AttentionCaptureCallback(L.Callback):
    def __init__(self, fps=10, n_batches_vis=10):
        """
        Args:
            fps: frames per second for the final animation
        """
        super().__init__()
        self.captured_attn = {}     # { layer_name: [attention tensor per batch] }
        self.captured_batches = []  # raw batch data if needed
        self.captured_outputs = []  # aggregator outputs for each batch
        self.fps = fps
        self.n_batches_vis = n_batches_vis

        self.sample_size_per_batch = []

        # We'll store references to highlight patches
        self.highlight_patch_x = None
        self.highlight_patch_y = None

    def setup(self, trainer, pl_module, stage):
        """
        Attach forward hooks to:
          - encoder cross-attn (enc_atn) => "encoder_cross_attn"
          - decoder cross-attn (dec_atn) => "decoder_cross_attn"
          - each process layer's self-attn => "process_layer_{i}"
        """
        attn_modules = []

        # If model has enc_atn/dec_atn
        if hasattr(pl_module.model, "enc_atn"):
            attn_modules.append((pl_module.model.enc_atn, "encoder_cross_attn"))
        if hasattr(pl_module.model, "dec_atn"):
            attn_modules.append((pl_module.model.dec_atn, "decoder_cross_attn"))

        # Add each process layer
        if hasattr(pl_module.model, "proc_layers"):
            for i, layer in enumerate(pl_module.model.proc_layers):
                # layer[0] is your RotarySelfAttention
                attn_modules.append((layer[0], f"process_layer_{i}"))

        def make_hook(layer_name):
            def hook_fn(module, inputs, outputs):
                # If forward returns (out, attn_weights)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    _, attn_weights = outputs
                else:
                    attn_weights = getattr(module, "last_attn_weights", None)

                if attn_weights is not None:
                    self.captured_attn.setdefault(layer_name, []).append(
                        attn_weights.detach().cpu()
                    )
            return hook_fn

        # Register hooks
        for (module, name) in attn_modules:
            module.register_forward_hook(make_hook(name))

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Store aggregator outputs + raw batch
        self.captured_outputs.append(outputs)
        self.captured_batches.append(batch)

    def on_test_end(self, trainer, pl_module):
        """
        Build one animation where:
          - total_frames = num_batches * batch_size
          - frame = (batch_idx, sample_idx)
          - update attention for that sample
          - only update the pred-vs-target lines when sample_idx == 0
        """
        if not self.captured_outputs:
            print("No outputs captured.")
            return
        if not self.captured_attn:
            print("No attention weights captured.")
            return

        # 1) Decide layer order and how many heads each layer has
        all_layers = list(self.captured_attn.keys())
        # e.g. "encoder_cross_attn", "process_layer_0", "decoder_cross_attn"
        encoder_layers = [n for n in all_layers if n == "encoder_cross_attn"]
        decoder_layers = [n for n in all_layers if n == "decoder_cross_attn"]
        proc_layers = sorted([n for n in all_layers if n.startswith("process_layer_")])
        layer_names = encoder_layers + proc_layers + decoder_layers

        # figure out heads for each layer
        layer_to_num_heads = {}
        for name in layer_names:
            # shape => (batch_size, heads, q_len, k_len)
            first_tensor = self.captured_attn[name][0]
            _, heads, _, _ = first_tensor.shape
            layer_to_num_heads[name] = heads

        max_heads = max(layer_to_num_heads.values())

        # 2) Make subplots
        # row=0 => pred vs target
        # rows=1.. => each layer
        rows = len(layer_names) + 1
        cols = max_heads
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        # (A) row=0: predictions vs. target
        pred_ax_x = axes[0][0]
        line_pred_x, = pred_ax_x.plot([], [], label="Pred X", color="blue", alpha=0.5)
        line_targ_x, = pred_ax_x.plot([], [], label="Targ X", color="cyan", alpha=0.5)
        pred_ax_x.legend()

        line_pred_y = None
        line_targ_y = None
        pred_ax_y = None
        if cols > 1:
            pred_ax_y = axes[0][1]
            line_pred_y, = pred_ax_y.plot([], [], label="Pred Y", color="red",    alpha=0.5)
            line_targ_y, = pred_ax_y.plot([], [], label="Targ Y", color="orange", alpha=0.5)
            pred_ax_y.legend()

        # Hide leftover columns in row=0
        for c in range(2, cols):
            axes[0][c].axis("off")

        # (B) row=1.. => each layer
        attn_images = []
        for i, layer_name in enumerate(layer_names):
            row_idx = i + 1
            heads_i = layer_to_num_heads[layer_name]

            first_attn_tensor = self.captured_attn[layer_name][0]  # shape (b, heads_i, q_len, k_len)
            _, _, q_len_i, k_len_i = first_attn_tensor.shape

            row_handles = []
            for col_j in range(cols):
                if col_j < heads_i:
                    im = axes[row_idx][col_j].imshow(
                        torch.zeros(q_len_i, k_len_i),
                        aspect='auto', cmap='viridis', vmin=0, vmax=1
                    )
                    axes[row_idx][col_j].axis("off")
                    axes[row_idx][col_j].set_title(f"{layer_name}, head={col_j}")
                    row_handles.append(im)
                else:
                    axes[row_idx][col_j].axis("off")
                    row_handles.append(None)
            attn_images.append(row_handles)

        plt.tight_layout()

        # 3) total_frames = num_batches * batch_size
        n_batches = len(self.captured_outputs)
        # find batch_size from first layer
        first_layer_first_batch = self.captured_attn[layer_names[0]][0]  # shape => (b, heads, q_len, k_len)
        bsize, _, _, _ = first_layer_first_batch.shape
        total_frames = min(n_batches, self.n_batches_vis) * bsize

        ## Compute number of samples per batch
        n_total_samples = 0
        for captured_output in self.captured_outputs:
            n_total_samples += len(captured_output.preds)
        
        total_frames = min(n_total_samples, total_frames)


        def remove_highlight_patches():
            """Remove existing highlight patches if present."""
            if self.highlight_patch_x is not None:
                self.highlight_patch_x.remove()
                self.highlight_patch_x = None
            if self.highlight_patch_y is not None:
                self.highlight_patch_y.remove()
                self.highlight_patch_y = None


        def update(global_frame_idx):
            """
            (batch_idx, sample_idx) = (global_frame_idx // bsize, global_frame_idx % bsize)
            - if sample_idx == 0 => update aggregator line plots
            - always update attention for the given sample
            """
            batch_idx = global_frame_idx // bsize
            sample_idx = global_frame_idx % bsize

            out_obj = self.captured_outputs[batch_idx]

            # If sample_idx==0, update aggregator-based line plots for that batch
            if sample_idx == 0:
                preds_list = out_obj.preds
                cat_preds_list = []
                readout_key = "cursor_velocity_2d"
                if readout_key not in out_obj.targets:
                    readout_key = "cursor_direction_to_target_2d"

                for wd in preds_list:
                    if readout_key in wd:
                        cat_preds_list.append(wd[readout_key])

                self.sample_size_per_batch = []
                for i_sample in range(len(preds_list)):
                    try:
                        self.sample_size_per_batch.append(preds_list[i_sample][readout_key].shape[0])
                    except:
                        self.sample_size_per_batch.append(0)

                target_tensor = out_obj.targets.get(readout_key, torch.zeros(0,2))
                has_events = (len(cat_preds_list) > 0) and (target_tensor.shape[0] > 0)

                if has_events:
                    concat_preds = torch.cat(cat_preds_list, dim=0)  # shape [N,2]
                    N = concat_preds.shape[0]
                    xs = range(N)
                    px = concat_preds[:, 0].cpu().numpy()
                    py = concat_preds[:, 1].cpu().numpy()

                    tx = target_tensor[:, 0].cpu().numpy()
                    ty = target_tensor[:, 1].cpu().numpy()

                    # X lines
                    line_pred_x.set_data(xs, px)
                    line_targ_x.set_data(xs, tx)
                    pred_ax_x.relim()
                    pred_ax_x.autoscale_view()
                    pred_ax_x.set_title(f"Batch={batch_idx} Pred vs. Target X")

                    # Y lines
                    if line_pred_y is not None and line_targ_y is not None:
                        line_pred_y.set_data(xs, py)
                        line_targ_y.set_data(xs, ty)
                        pred_ax_y.relim()
                        pred_ax_y.autoscale_view()
                        pred_ax_y.set_title(f"Batch={batch_idx} Pred vs. Target Y")
                else:
                    # no events => clear lines
                    line_pred_x.set_data([], [])
                    line_targ_x.set_data([], [])
                    pred_ax_x.set_title(f"No aggregator events: batch={batch_idx}")
                    pred_ax_x.relim()
                    pred_ax_x.autoscale_view()

                    if line_pred_y is not None and line_targ_y is not None:
                        line_pred_y.set_data([], [])
                        line_targ_y.set_data([], [])
                        pred_ax_y.set_title(f"No aggregator events: batch={batch_idx}")
                        pred_ax_y.relim()
                        pred_ax_y.autoscale_view()

            # Highlight the current samples
            start = np.sum(self.sample_size_per_batch[:sample_idx])
            end = start + self.sample_size_per_batch[sample_idx]

            # remove old patches
            remove_highlight_patches()

            # add new patches
            # highlight X:
            self.highlight_patch_x = pred_ax_x.axvspan(start, end, color='gray', alpha=0.2)

            # highlight Y if we have a second axis
            if pred_ax_y is not None:
                self.highlight_patch_y = pred_ax_y.axvspan(start, end, color='gray', alpha=0.2)


            # (C) ALWAYS update attention for that sample
            for row_i, layer_name in enumerate(layer_names):
                row_idx = row_i + 1
                attn_for_that_batch = self.captured_attn[layer_name][batch_idx]
                # shape => (bsize, heads, q_len, k_len)
                attn_for_sample = attn_for_that_batch[sample_idx]  # shape => (heads, q_len, k_len)

                heads_i = layer_to_num_heads[layer_name]
                for col_j in range(cols):
                    if col_j < heads_i:
                        attn_map = attn_for_sample[col_j].numpy()
                        attn_images[row_i][col_j].set_data(attn_map)
                        attn_images[row_i][col_j].autoscale()

            return []

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000/self.fps,  # e.g. fps=2 => 500ms
            blit=False,
            repeat=False
        )

        my_writer = animation.FFMpegWriter(
            fps=self.fps,
            codec="h264",
            bitrate=-1,        # Or some chosen bitrate
            extra_args=["-threads", "24"]  # <--- forces ffmpeg to use 8 threads
        )

        # plt.show()
        # or:
        ani.save("pred_vs_target_plus_all_samples_attn.mp4", writer=my_writer, dpi=100)



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
    attn_callback = AttentionCaptureCallback()

    # 7) Build trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        # callbacks=[evaluator],  # keep it minimal
        callbacks=[evaluator, attn_callback],  # keep it minimal
        # any other trainer configs
    )

    # 8) Run test
    trainer.test(wrapper, datamodule=test_data_module)



if __name__ == "__main__":
    main()
