import os

from torch.distributed import init_process_group


def ddp_setup(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
