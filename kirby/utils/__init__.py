import torch

from .dir_utils import find_files_by_extension, make_directory
from .distributed import ddp_setup
from .logging import logging
from .seed_everything import seed_everything


def move_to(data: dict, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
        elif isinstance(data[key], dict):
            data[key] = move_to(data[key], device)
    return data
