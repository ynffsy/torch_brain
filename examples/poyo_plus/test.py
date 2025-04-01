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
    def __init__(self):
        super().__init__()
        self.captured_attn = {}      # dict of {layer_name: [list_of_tensors_per_batch]}
        self.captured_batches = []   # store the actual batch data if needed
        self.captured_outputs = []  # store the actual outputs if needed


    def setup(self, trainer, pl_module, stage):
        """Attach forward hooks, etc., just like before."""
        ## Add all attention modules
        attn_modules = []
        for i, layer in enumerate(pl_module.model.proc_layers):
            attn_modules.append(layer[0])  # e.g. RotarySelfAttention


        def make_hook(layer_name):
            def hook_fn(module, inputs, outputs):
                # If your module returns (out, attn_weights):
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    out, attn_weights = outputs
                else:
                    # Or if module.last_attn_weights is set:
                    attn_weights = getattr(module, "last_attn_weights", None)

                if attn_weights is not None:
                    # store the attn in a list
                    self.captured_attn.setdefault(layer_name, []).append(attn_weights.detach().cpu())
            return hook_fn

        for i, layer in enumerate(attn_modules):
            layer_name = f"attention_layer_{i}"
            layer.register_forward_hook(make_hook(layer_name))


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        We'll store the 'batch' here so we can reference it later for visualization.
        'batch' might be a dict or a tuple depending on your DataLoader.
        """
        self.captured_batches.append(batch)
        self.captured_outputs.append(outputs)


    def on_test_end(self, trainer, pl_module):
        """
        Called after all test batches are done.
        We'll create a single animation over 'n_frames' = the number of test batches.
        For each batch:
          - We plot the concatenated predictions vs. targets for 'cursor_direction_to_target_2d'.
          - We display attention maps for each layer & head (just picking sample=0 in the batch).
        """

        ## Sanity check: do we have data?
        if not len(self.captured_outputs):
            print("No outputs captured.")
            return
        if not len(self.captured_attn):
            print("No attention weights captured.")
            return

        layer_names = list(self.captured_attn.keys())
        n_layers = len(layer_names)
        # e.g. shape => [batch_size, n_heads, seq_len, seq_len]
        first_attn = self.captured_attn[layer_names[0]][0]
        batch_size, n_heads, seq_len, _ = first_attn.shape

        # The number of frames in our animation = number of test batches we captured
        n_frames = len(self.captured_outputs)  # or len(self.captured_attn[layer_names[0]])
        
        ## Create the figure and axes grid
        rows = n_layers + 1  # row0 for preds, row1..n_layers for attention
        cols = n_heads       # each column is a different head
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

        # If there's only 1 layer or 1 head, sometimes axes is not 2D.
        # Let's ensure it's always a 2D list.
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            # Then each row is just one axis
            axes = [[ax] for ax in axes]
        
        ## predictions vs. targets plots
        pred_ax_x = axes[0][0]
        pred_ax_y = axes[0][1]

        pred_ax_x.set_title("Pred vs. Target (x)")
        pred_ax_y.set_title("Pred vs. Target (y)")

        # We'll create 4 line objects: pred_x, targ_x, pred_y, targ_y
        line_pred_x, = pred_ax_x.plot([], [], label="Pred X", color="blue")
        line_targ_x, = pred_ax_x.plot([], [], label="Targ X", color="cyan")

        line_pred_y, = pred_ax_y.plot([], [], label="Pred Y", color="red")
        line_targ_y, = pred_ax_y.plot([], [], label="Targ Y", color="orange")

        pred_ax_x.legend()
        pred_ax_y.legend()


        # Hide other columns in row=0 (if n_heads>1)
        for c in range(2, cols):
            axes[0][c].axis("off")

        ## Rows=1..n_layers: attention maps
        # We'll store an imshow handle for each layer & head
        attn_images = []
        for layer_i in range(n_layers):
            row_i = layer_i + 1  # 1-based offset
            row_handles = []
            for head_j in range(n_heads):
                im = axes[row_i][head_j].imshow(torch.zeros(seq_len, seq_len),
                                                aspect='auto', cmap='viridis', vmin=0, vmax=1)
                axes[row_i][head_j].set_title(f"{layer_names[layer_i]} head {head_j}")
                axes[row_i][head_j].axis("off")
                row_handles.append(im)
            attn_images.append(row_handles)

        plt.tight_layout(pad=3)

        ## Define the update function for animation
        def update(frame_idx):
            """
            Called for each test batch (frame_idx).
              1) Retrieve predictions from self.captured_outputs[frame_idx], 
                 handle the case of zero-length or missing predictions.
              2) Update the pred-vs-target lines on pred_ax_x/pred_ax_y
              3) Update attention for each layer/head
            """

            out_obj = self.captured_outputs[frame_idx]
            preds_list = out_obj.preds  # list of sub-window dicts

            ## Gather & concatenate predictions
            cat_preds_list = []
            for window_dict in preds_list:
                # If there's no "cursor_direction_to_target_2d" in this sub-window, skip it
                if "cursor_direction_to_target_2d" in window_dict:
                    cat_preds_list.append(window_dict["cursor_direction_to_target_2d"])

            # If either we have no sub-windows or the aggregator's target is empty, handle it
            target_tensor = out_obj.targets["cursor_direction_to_target_2d"]  # shape e.g. [N, 2] or [0, 2]

            # Check if we actually have any predictions or if targets are empty
            has_events = (len(cat_preds_list) > 0) and (target_tensor.shape[0] > 0)

            if has_events:
                # Concatenate partial predictions
                concat_preds = torch.cat(cat_preds_list, dim=0)  # e.g. shape [N,2]
                N = concat_preds.shape[0]

                xs = range(N)
                px = concat_preds[:, 0].cpu().numpy()
                py = concat_preds[:, 1].cpu().numpy()

                tx = target_tensor[:, 0].cpu().numpy()
                ty = target_tensor[:, 1].cpu().numpy()

                # Update lines for X
                line_pred_x.set_data(xs, px)
                line_targ_x.set_data(xs, tx)
                pred_ax_x.relim()
                pred_ax_x.autoscale_view()
                pred_ax_x.set_title(f"Pred vs. Target X (batch={frame_idx})")

                # Update lines for Y
                line_pred_y.set_data(xs, py)
                line_targ_y.set_data(xs, ty)
                pred_ax_y.relim()
                pred_ax_y.autoscale_view()
                pred_ax_y.set_title(f"Pred vs. Target Y (batch={frame_idx})")

            else:
                # No valid events => clear the lines
                line_pred_x.set_data([], [])
                line_targ_x.set_data([], [])
                line_pred_y.set_data([], [])
                line_targ_y.set_data([], [])

                pred_ax_x.set_title(f"No events in batch={frame_idx}")
                pred_ax_y.set_title(f"No events in batch={frame_idx}")

                # It's fine to keep re-lim and autoscale (they won't change anything).
                pred_ax_x.relim()
                pred_ax_x.autoscale_view()
                pred_ax_y.relim()
                pred_ax_y.autoscale_view()

            ## Update attention maps
            for layer_i, layer_name in enumerate(layer_names):
                # shape => [batch_size, heads, seq_len, seq_len]
                attn_for_batch = self.captured_attn[layer_name][frame_idx]
                # pick the first sample in that batch
                attn_for_sample = attn_for_batch[0]

                for head_j in range(n_heads):
                    attn_map = attn_for_sample[head_j].cpu().numpy()
                    attn_images[layer_i][head_j].set_data(attn_map)
                    attn_images[layer_i][head_j].autoscale()

            return []

        ## Build the animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,   # go through each test batch
            interval=1000,     # ms between frames (1 FPS)
            blit=False,
            repeat=False
        )

        plt.show()
        # ani.save("pred_attn_animation.gif", writer="imagemagick", fps=1)



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
        callbacks=[evaluator, attn_callback],  # keep it minimal
        # any other trainer configs
    )

    # 8) Run test
    trainer.test(wrapper, datamodule=test_data_module)



if __name__ == "__main__":
    main()
