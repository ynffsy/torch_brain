import os
from tqdm import tqdm
import argparse
from hydra import compose, initialize
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import logging
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from model import (
    BhvrDecoder,
    ContextManager,
    Encoder,
    NDT2Model,
    SpikesPatchifier,
)
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from transforms import Ndt2Tokenizer
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import SequentialFixedWindowSampler
from train import TrainWrapper

# from scripts.eval_utils import bin_behaviors, viz_single_cell

# import logging

# logging.basicConfig(level=logging.INFO)

# torch.set_float32_matmul_precision("medium")


def move_to_gpu(d, device):
    for k, v in d.items():
        if isinstance(v, dict):
            move_to_gpu(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)


def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:, :, i] = arr == _uni
    return ret


def get_ctx_vocab(self, ctx_keys):
    return {k: getattr(self.dataset, f"get_{k}_ids")() for k in ctx_keys}


# -------
# SET UP
# -------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="d23a44ef-1402-4ed7-97f5-47e9a7a504d9")
ap.add_argument("--behavior", type=str, default="choice")
ap.add_argument("--model_path", type=str, default="./logs/lightning_logs/")
ap.add_argument("--ckpt_name", type=str, default="/home/hice1/aandre8/scratch/alex/checkpoints/")
ap.add_argument("--save_path", type=str, default="./results/")
ap.add_argument("--config_path", type=str, default="./configs/")
ap.add_argument("--config_name", type=str, default="probe_choice.yaml")
args = ap.parse_args()

save_path = args.save_path
config_path = args.config_path
model_path = args.model_path
config_name = args.config_name
ckpt_name = args.ckpt_name
eid = args.eid
bhvr = args.behavior

logging.info(f"Evaluating session: {eid}")

params = {
    "interval_len": 2,
    "binsize": 0.02,
    "single_region": False,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
    "fr_thresh": 0.5,
}

with initialize(version_base="1.3", config_path="./ibl_configs"):
    cfg = compose(config_name=config_name)


# ---------
# LOAD DATA
# ---------
dataset_cfg = OmegaConf.create([
    {
        "selection": [
            {
                "brainset": "ibl",
                "sessions": [eid]
            }
        ]
    }
])

dataset = Dataset(root=cfg.data_root, split="test", config=dataset_cfg)

# ----------
# LOAD MODEL
# ----------

L.seed_everything(cfg.seed)
# seed_everything(cfg.seed)


with open_dict(cfg):
    # Adjust batch size for multi-gpu
    num_gpus = torch.cuda.device_count()
    cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
    cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
    cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
#     log.info(f"Number of GPUs: {num_gpus}")
#     log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
#     log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

# wandb_logger = set_wandb(cfg, log)
wandb_logger = None

dim = cfg.model.dim

# Mask manager (for MAE SSL)
mae_mask_manager = None

# Context manager
ctx_manager = ContextManager(dim, cfg.ctx_keys)

# Spikes patchifier
spikes_patchifier = SpikesPatchifier(dim, cfg.patch_size)

# Encoder
encoder = Encoder(
    dim=dim,
    max_time_patches=cfg.model.max_time_patches,
    max_space_patches=cfg.model.max_space_patches,
    **cfg.model.encoder,
)

# Decoder
decoder = BhvrDecoder(
    dim=dim,
    max_time_patches=cfg.model.max_time_patches,
    max_space_patches=cfg.model.max_space_patches,
    bin_time=cfg.bin_time,
    **cfg.model.bhv_decoder,
)

# Model wrap everithing
model = NDT2Model(mae_mask_manager, ctx_manager, spikes_patchifier, encoder, decoder)


# ckpt = torch.load(f"{model_path}/{ckpt_name}/last.ckpt", map_location="cpu")


bhvr_dim = cfg.model.bhv_decoder["behavior_dim"]

# Load from checkpoint
# TODO update this
path = f"{cfg.checkpoint_path}/probe_{bhvr}-{eid}.ckpt"
ckpt = torch.load(path)
model.ctx_manager.load_state_dict(ckpt["context_manager_state_dict"])
model.spikes_patchifier.load_state_dict(ckpt["spikes_patchifier_state_dict"])
model.encoder.load_state_dict(ckpt["encoder_state_dict"])
model.decoder.load_state_dict(ckpt["decoder_state_dict"])

ctx_tokenizer = ctx_manager.get_ctx_tokenizer()
tokenizer = Ndt2Tokenizer(
    ctx_time=cfg.ctx_time,
    bin_time=cfg.bin_time,
    patch_size=cfg.patch_size,
    pad_val=cfg.pad_val,
    ctx_tokenizer=ctx_tokenizer,
    unsorted=cfg.unsorted,
    is_ssl=cfg.is_ssl,
    bhvr_key=cfg.get("bhvr_key"),
    bhvr_dim=bhvr_dim,
    ibl_binning=cfg.get("ibl_binning", False),
)

test_dataset = Dataset(
    root=cfg.data_root, config=dataset_cfg, split="test", transform=tokenizer
)
inter = test_dataset.get_sampling_intervals()
eval_sampler = SequentialFixedWindowSampler(
    interval_dict=inter, window_length=cfg.ctx_time, drop_short=True
)

dataset = Dataset(
    root=cfg.data_root, config=dataset_cfg, split=None, transform=tokenizer
)
batch_size = 8
eval_loader = DataLoader(
    dataset=dataset,
    # batch_size=cfg.batch_size,
    batch_size = batch_size,
    sampler=eval_sampler,
    collate_fn=collate,
    num_workers=cfg.num_workers,
)

# TODO CHECK THIS
# sequence_length = 1.0
# val_sampler = SequentialFixedWindowSampler(
#     interval_dict=val_dataset.get_sampling_intervals(),
#     window_length=sequence_length,
#     step=sequence_length / 2,
# )

# optimizer = Lamb(
#     model.parameters(),  # filter(lambda p: p.requires_grad, model.parameters()),
#     lr=max_lr,
#     weight_decay=cfg.weight_decay,
# )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=max_lr,
#     epochs=epochs,
#     steps_per_epoch=len(val_loader),
#     pct_start=cfg.pct_start,
#     anneal_strategy="cos",
#     div_factor=1,
# )

# wrapper = train_wrapper.TrainWrapper(
#     model=model,
#     optimizer=optimizer,
#     scheduler=scheduler,
# )


# Callbacks
callbacks = [ModelSummary(max_depth=3)]

# Set up trainer
trainer = L.Trainer(
    logger=wandb_logger,
    default_root_dir=cfg.log_dir,
    check_val_every_n_epoch=cfg.eval_epochs,
    max_epochs=cfg.epochs,
    log_every_n_steps=1,
    callbacks=callbacks,
    accelerator="gpu",
    precision=cfg.precision,
    num_sanity_val_steps=0,
    strategy="auto",
)

# Train wrapper
# TODO not confident here
train_wrapper = TrainWrapper(cfg, model)


# ---------
# INFERENCE
# ---------
session_timestamp = {}
session_subtask_index = {}
session_gt_output = {}
session_pred_output = {}

for batch in tqdm(eval_loader):
    # absolute_starts = batch.pop("absolute_start")  # (B,)
    # session_ids = batch.pop("session_id")  # (B,)
    # output_subtask_index = batch.pop("output_subtask_index")

    # batch_format = None
    # if "input_mask" in batch:
    #     batch_format = "padded"
    # elif "input_seqlen" in batch:
    #     batch_format = "chained"
    # else:
    #     raise ValueError("Invalid batch format.")

    # # move_to_gpu(batch, pl_module)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # move_to_gpu(batch, device)

    # Autocast is explicitly set based on the precision specified by the user.
    # By default, torch autocasts to float16 for 16-bit inference.
    # This behavior is overridden to use bfloat16 if specified in trainer.precision.
    # If 16-bit inference is not enabled, autocast is not used.
    def get_autocast_args(trainer):
        if trainer.precision.startswith("bf16"):
            return torch.bfloat16, True
        elif trainer.precision.startswith("16"):
            return torch.float16, True
        else:
            return None, False

    dtype, enabled = get_autocast_args(trainer)
    
    with torch.inference_mode():
        decoder_out = model(batch, "bhv")

    decoder_out = model(batch, "bhv")

    # # forward pass
    # with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
    #     with torch.inference_mode():
    #         # pred_output, loss, losses_taskwise = model(**batch)
    #         decoder_out = model(batch, "bhv")
    
    print(decoder_out)

    print("exit")
    exit(0)

    # we need to get the timestamps, the ground truth values, the task ids as well
    # as the subtask ids. since the batch is padded and chained, this is a bit tricky
    # tldr: this extracts the ground truth in the same format as the model output
    batch_size = len(pred_output)
    # get gt_output and timestamps to be in the same format as pred_output
    timestamps = [{} for _ in range(batch_size)]
    subtask_index = [{} for _ in range(batch_size)]
    gt_output = [{} for _ in range(batch_size)]

    # collect ground truth
    for taskname, spec in model.readout.decoder_specs.items():
        taskid = Decoder.from_string(taskname).value

        # get the mask of tokens that belong to this task
        mask = batch["output_decoder_index"] == taskid

        if not torch.any(mask):
            # there is not a single token for this task, so we skip
            continue

        # we need to distribute the outputs to their respective samples

        if batch_format == "padded":
            token_batch = torch.where(mask)[0]
        elif batch_format == "chained":
            token_batch = batch["output_batch_index"][mask]

        batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
        for i in range(len(batch_i)):
            timestamps[batch_i[i]][taskname] = (
                batch["output_timestamps"][mask][token_batch == i]
                + absolute_starts[batch_i[i]]
            )
            subtask_index[batch_i[i]][taskname] = output_subtask_index[taskname][
                (token_batch == i).detach().cpu()
            ]
            gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                token_batch == i
            ]

    # register all of the data
    for i in range(batch_size):
        session_id = session_ids[i]

        if session_id not in session_pred_output:
            session_pred_output[session_id] = {}
            session_gt_output[session_id] = {}
            session_timestamp[session_id] = {}
            session_subtask_index[session_id] = {}

        for taskname, pred_values in pred_output[i].items():
            if taskname not in session_pred_output[session_id]:
                session_pred_output[session_id][taskname] = pred_values.detach().cpu()
                session_gt_output[session_id][taskname] = (
                    gt_output[i][taskname].detach().cpu()
                )
                session_timestamp[session_id][taskname] = (
                    timestamps[i][taskname].detach().cpu()
                )
                session_subtask_index[session_id][taskname] = (
                    subtask_index[i][taskname].detach().cpu()
                )
            else:
                session_pred_output[session_id][taskname] = torch.cat(
                    (
                        session_pred_output[session_id][taskname],
                        pred_values.detach().cpu(),
                    )
                )
                session_gt_output[session_id][taskname] = torch.cat(
                    (
                        session_gt_output[session_id][taskname],
                        gt_output[i][taskname].detach().cpu(),
                    )
                )
                session_timestamp[session_id][taskname] = torch.cat(
                    (
                        session_timestamp[session_id][taskname],
                        timestamps[i][taskname].detach().cpu(),
                    )
                )
                session_subtask_index[session_id][taskname] = torch.cat(
                    (
                        session_subtask_index[session_id][taskname],
                        subtask_index[i][taskname].detach().cpu(),
                    )
                )

# -----
# EVAL
# -----
results = {"Choice": {}, "Block": {}, "Whisker": {}, "Wheel": {}}

# Eval discrete behaviors
session_id = f"ibl_{eid}/{eid}"
test_data = dataset.get_session_data(session_ids[0])
intervals = np.c_[
    dataset.get_session_data(session_id).trials.start,
    dataset.get_session_data(session_id).trials.end,
]
choice = dataset.get_session_data(session_id).choice.choice
block = dataset.get_session_data(session_id).block.block
reward = dataset.get_session_data(session_id).reward.reward

if args.behavior == "choice":
    choice = session_gt_output[session_id]["CHOICE"]
    pred = session_pred_output[session_id]["CHOICE"].argmax(-1)
    results["Choice"]["accuracy"] = accuracy_score(choice, pred)
    results["Choice"]["balanced_accuracy"] = balanced_accuracy_score(choice, pred)

if args.behavior == "block":
    block = session_gt_output[session_id]["BLOCK"]
    pred = session_pred_output[session_id]["BLOCK"].argmax(-1)
    results["Block"]["accuracy"] = accuracy_score(block, pred)
    results["Block"]["balanced_accuracy"] = balanced_accuracy_score(block, pred)

# Eval continuous behaviors
if args.behavior == "wheel":
    wh_gt = session_gt_output[session_id]["WHEEL"]
    wh_pred = session_pred_output[session_id]["WHEEL"]
    wh_timestamps = session_timestamp[session_id]["WHEEL"]
    wh_subtask_index = session_subtask_index[session_id]["WHEEL"]

    wh_gt_vals = wh_gt.squeeze()
    wh_pred_vals = wh_pred.squeeze()

    behave_dict, mask_dict = bin_behaviors(
        wh_timestamps, wh_gt_vals.numpy(), intervals=intervals, beh="wheel", **params
    )

    binned_wh_gt = behave_dict["wheel"]

    behave_dict, mask_dict = bin_behaviors(
        wh_timestamps, wh_pred_vals.numpy(), intervals=intervals, beh="wheel", **params
    )
    binned_wh_pred = behave_dict["wheel"]

if args.behavior == "whisker":
    me_gt = session_gt_output[session_id]["WHISKER"]
    me_pred = session_pred_output[session_id]["WHISKER"]
    me_timestamps = session_timestamp[session_id]["WHISKER"]
    me_subtask_index = session_subtask_index[session_id]["WHISKER"]

    me_gt_vals = me_gt.squeeze()
    me_pred_vals = me_pred.squeeze()

    behave_dict, mask_dict = bin_behaviors(
        me_timestamps, me_gt_vals.numpy(), intervals=intervals, beh="whisker", **params
    )

    binned_me_gt = behave_dict["whisker"]

    behave_dict, mask_dict = bin_behaviors(
        me_timestamps,
        me_pred_vals.numpy(),
        intervals=intervals,
        beh="whisker",
        **params,
    )
    binned_me_pred = behave_dict["whisker"]

if (args.behavior == "whisker") or (args.behavior == "wheel"):
    T = 100
    X = np.concatenate(
        (
            _one_hot(choice.reshape(-1, 1), T),
            _one_hot(reward.reshape(-1, 1), T),
            _one_hot(block.reshape(-1, 1), T),
        ),
        axis=2,
    )

    var_name2idx = {"block": [2], "choice": [0], "reward": [1]}
    var_value2label = {
        "block": {
            (0,): "p(left)=0.2",
            (1,): "p(left)=0.5",
            (2,): "p(left)=0.8",
        },
        "choice": {(0,): "right", (1,): "left"},
        "reward": {
            (0,): "no reward",
            (1,): "reward",
        },
    }
    var_tasklist = ["block", "choice", "reward"]
    var_behlist = []

    if args.behavior == "wheel":
        gt = binned_wh_gt.reshape(-1, T, 1)
        pred = binned_wh_pred.reshape(-1, T, 1)
        avail_beh = "wheel-speed"
    elif args.behavior == "whisker":
        gt = binned_me_gt.reshape(-1, T, 1)
        pred = binned_me_pred.reshape(-1, T, 1)
        avail_beh = "whisker-motion-energy"

    y = gt[:, :, [0]]
    y_pred = pred[:, :, [0]]
    _r2_psth, _r2_trial = viz_single_cell(
        X,
        y,
        y_pred,
        var_name2idx,
        var_tasklist,
        var_value2label,
        var_behlist,
        subtract_psth="task",
        aligned_tbins=[],
        neuron_idx=avail_beh,
        neuron_region="",
        method="poyo",
        save_path="../results/",
        save_plot=False,
    )
    plt.close("all")
    beh_name = "Wheel" if avail_beh == "wheel-speed" else "Whisker"
    results[beh_name]["r2_psth"] = _r2_psth
    results[beh_name]["r2_trial"] = _r2_trial

    # res_path = f"{save_path}/{eid}/"
    # os.makedirs(res_path, exist_ok=True)
    save_res = {"gt": y, "pred": y_pred, "beh_name": beh_name, "eid": eid}
    # os.makedirs(f"{save_path}/raw/", exist_ok=True)
    # np.save(f"{save_path}/raw/{eid}_{beh_name}.npy", save_res)

print(results)

res_path = f"{save_path}/{eid}/"
if not os.path.exists(res_path):
    os.makedirs(res_path)
np.save(f"{res_path}/{args.behavior}.npy", results)
