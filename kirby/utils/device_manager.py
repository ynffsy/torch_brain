import os

import torch
import torch.distributed as dist
from absl import flags

from kirby.utils import logging


log = logging(header='DEVICE', header_color='gray')

# Settings for CPU and multi-GPU training.
flags.DEFINE_integer('num_workers', 0, 'Number of CPU workers for training.')
flags.DEFINE_string('port', '13355', 'Port for master GPU.')
flags.DEFINE_boolean('no_cuda', False, 'disables CUDA training')
FLAGS = flags.FLAGS


class DeviceManager:
    r"""Handles setting up cpu, gpu or multi-gpu devices.

    Args:
        no_cuda (bool, Optional): Whether to use a cpu device for training. (default: :obj:`False`)
        distributed (bool, Optional): Whether the training will be distributed. :obj:`no_cuda` and :obj:`distributed`
            are exclusive. (default: :obj:`False`)
        world_size (int, Optional): Number of devices used for training, required only if :obj:`distributed` is set to
            :obj:`True`.
        rank (int, Optional): Device rank, required only if :obj:`distributed` is set to :obj:`True`.
        port (str, Optional): Group port used for distributed process, required only if :obj:`distributed` is set
            to :obj:`True`.

    In cpu or gpu mode, :attr:`world_size` will be set to :obj:`1` and :attr:`rank` to :obj:`0`. Access to distributed
    mode specific attributes will raise an error.

    .. note::
        The master device will have rank 0.
    """
    @property
    def rank(self) -> int:
        r"""Device rank."""
        return self.__rank

    @property
    def distributed(self) -> bool:
        r"""Whether training will use multiple devices."""
        return self.__distributed

    @property
    def world_size(self) -> int:
        r"""Number of devices."""
        return self.__world_size

    @property
    def device(self) -> torch.device:
        r"""Device."""
        if self.__device is None:
            raise AttributeError('Device has not been initialized yet.')
        return self.__device

    @property
    def group(self) -> torch.distributed.group:
        r"""Distributed group. Only available in distributed mode."""
        if not self.distributed:
            raise AttributeError('Not in distributed mode.')
        if self.__group is None:
            # todo not attributeerror
            raise AttributeError('Group has not been initialized yet.')
        return self.__group

    def __init__(self):
        # default
        self.__world_size = None
        self.__rank = None
        self.__group = None
        self.__device = None

    def setup(self, *, no_cuda=False, distributed=False, world_size=None, rank=None, port=None):
        no_cuda = no_cuda or FLAGS.no_cuda
        port = port or FLAGS.port
        
        # Setup CPU and GPU
        assert not(distributed and no_cuda), 'Both flags for CPU-training and Multi-GPU training were set to True.'
        self.__distributed = distributed

        if no_cuda:
            self.__device = self.__setup_no_cuda()
        elif not self.distributed:
            self.__device = self.__setup_cuda()
        else:
            self.__world_size = world_size
            self.__rank = rank
            self.__device, self.__group = self.__setup_distributed(port)
        return self.device

    def __setup_no_cuda(self):
        # use cpu
        log.info('Using CPU Device.')
        return torch.device('cpu')

    def __setup_cuda(self):
        # make sure at least one gpu is available.
        assert torch.cuda.is_available(), 'No cuda devices found.'

        # count number of visible devices
        num_devices = torch.cuda.device_count()
        log.debug('Found {} CUDA Devices{}.'.format(num_devices, ', will only use first '
                                                                              'device.' if num_devices > 1 else ''))

        # select first visible gpu
        device = torch.device(f'cuda')
        log.info('Single GPU {} was selected.'.format(device))

        return device

    def __setup_distributed(self, port):
        assert self.world_size > 1, 'Cannot run distributed training with world_size={}.'.format(self.world_size)
        assert self.rank is not None, 'Rank not provided, Multi-GPU not setup correctly.'
        assert port is not None, 'Port for distributed training not provided.'

        # verify that there are enough gpus.
        num_devices = torch.cuda.device_count()
        assert num_devices >= self.world_size, 'Not enough GPUs, requested {} gpus, but only {} are ' \
                                               'visible.'.format(self.world_size, num_devices)

        # select device
        device = torch.device(f'cuda:{self.rank}')
        log.info('(world size {}, master rank {}) '
                 '{} has rank {}.'.format(self.world_size, 0, device, self.rank))
        torch.cuda.set_device(self.rank)

        # setup dist process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        # todo does this need to be run in each node or only in node with rank 0? same for new_group
        dist.init_process_group(backend='nccl', init_method='env://', rank=self.rank, world_size=self.world_size)
        group = dist.new_group()
        log.debug('(rank:{}), group initialize, port {}.'.format(self.rank, port))

        return device, group
