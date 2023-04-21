import os
import shutil

from absl import flags
import torch

from kirby.utils import logging


log = logging(header='CHECKPOINT', header_color='cyan')

flags.DEFINE_string('initial_checkpoint', None, 'The initial checkpoint in .pt format.')
flags.DEFINE_integer('checkpoint_epochs', 10, 'Save checkpoint at at every checkpoint_epochs.')


class CheckpointManager:
    r"""Checkpointer that provides simple :meth:`save_checkpoint` and :meth:`resume_from_checkpoint` utilities.
    """
    def __init__(self, logdir=None, ckpt_filename_format="ckpt-{}.pt", save_every=1, save_latest=True, rank=0):
        # checkpoint pattern
        self.logdir = logdir

        self.save_every = save_every 
        self.rank = rank
        if logdir is not None:
            self.resumed_from_checkpoint = False
            self.__ckpt_path_format = os.path.join(self.logdir, ckpt_filename_format)
            self.latest_ckpt_path = self.__ckpt_path_format.format('latest')

    def extract_model_core(self, model):
        r"""Extracts the core model from a distributed model.
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model

    def save_checkpoint(self, model, *, optimizer=None, epoch: int=None, **kwargs):
        r"""Saves checkpoint to path, only for the master device.

        Args:
            epoch (int): Training epoch, used as suffix.

        .. note::
            The recommended way of saving a checkpoint from a distributed model is to save the core module's state.
        """
        if epoch is not None and epoch % self.save_every != 0:
            return

        if self.rank != 0:
            return
        path = self.__ckpt_path_format.format(epoch) if epoch is not None else self.__ckpt_path_format.format('latest')
        model_core = self.extract_model_core(model)

        state = {'model': model_core.state_dict(),
                 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                 'epoch': epoch,
                 **kwargs}

        torch.save(state, path)

        shutil.copy(path, self.latest_ckpt_path)
        log.debug('Model (epoch: {}) saved.'.format(epoch))


    def drop_modules(self, state_dict, drop_list):
        r"""Drops modules in drop_list from state_dict.
        """

        new_state_dict = {}

        for key in state_dict.keys():
            if any(key.startswith(drop) for drop in drop_list):
                continue
            new_state_dict[key] = state_dict[key]
        return new_state_dict

    def resume_from_checkpoint(self, resume_ckpt, model, device=None, strict=True, optimizer=None, drop_list=None):
        r"""Loads model weights from :obj:`resume_ckpt`.

        Args:
            resume_ckpt (String): Path to checkpoint.
            strict (bool): Whether to strictly enforce that the keys in current and checkpointed modules match.
                (default: :obj:`True`)

        .. note::
            Performs a checksum test to make sure the same hyperparameters were used during training.
        """
        # load checkpoint
        log.info('Loading model from {}.'.format(resume_ckpt))
        device = device if device is not None else torch.device('cpu')
        checkpoint = torch.load(resume_ckpt, map_location=device)

        # load model state
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel), \
            "Weights need to be loaded before model is distributed."
        
        state_dict = checkpoint['model']
        if drop_list is not None:
            state_dict = self.drop_modules(state_dict, drop_list)
        model.load_state_dict(state_dict, strict=strict and drop_list is None)

        # load optimizier state
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info('Loaded optimizer state.')

        self.resumed_from_checkpoint = True
        return checkpoint
