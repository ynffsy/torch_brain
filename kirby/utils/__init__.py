from .logging import logging
from .device_manager import DeviceManager
from .dir_utils import find_files_by_extension, make_directory
from .distributed import ddp_setup
from .seed_everything import seed_everything
from .checkpointer import CheckpointManager

import torch


def move_to(data: dict, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
