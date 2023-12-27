import pickle

old_unpickler = pickle.Unpickler  # Unfortunate hack to fix a bug in Lightning.
# https://github.com/Lightning-AI/lightning/issues/18152
# Will likely be fixed by 2.1.0.
import lightning

pickle.Unpickler = old_unpickler

from collections import OrderedDict

import hydra
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

# Flags are absorbed by Hydra.
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_optimizer import Lamb

from kirby.data import Collate, Dataset, build_vocab
from kirby.tasks.reaching import REACHING
from kirby.taxonomy import decoder_registry, weight_registry
from kirby.transforms import Compose
from kirby.utils import logging, seed_everything, train_wrapper


def run_training(cfg: DictConfig):
    # Fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # Higher speed on machines with tensor cores.
    torch.set_float32_matmul_precision("medium")

    logging.init_logger()
    log = logging.getLogger()

    # Device setup is managed by PyTorch Lightning.
    # prepare transform
    # The transform list is defined in the config file.
    sequence_length = 1.0
    transforms = hydra.utils.instantiate(
        cfg.train_transforms, sequence_length=sequence_length
    )

    transform = Compose(transforms)
    log.info("Data root: {}".format(cfg.data_root))
    train_dataset = Dataset(
        cfg.data_root,
        "train",
        include=cfg.train_datasets,
        transform=transform,
    )
    # In Lightning, testing only happens once, at the end of training. To get the
    # intended behavior, we need to specify a validation set instead.
    val_dataset = Dataset(
        cfg.data_root,
        "test",
        include=cfg.val_datasets,
    )

    # Build a vocabulary from these unit names.
    vocab = build_vocab(train_dataset.unit_names, val_dataset.unit_names)

    log.info(f"Number of training samples: {len(train_dataset)}")
    log.info(f"Number of sessions: {len(train_dataset.session_names)}")
    log.info(f"Number of units: {len(train_dataset.unit_names)}")

    # Make model
    # Note _convert_ is set to object otherwise decoder_registry gets cast to
    # a DictConfig and this interferes with checkpointing down the line.
    # We add the unit vocabulary here so it gets saved automatically, and the weights
    # are associated with their vocabulary.
    model = hydra.utils.instantiate(
        cfg.model,
        unit_vocab=vocab,
        session_names=train_dataset.session_names,
        task_specs=decoder_registry,
        _convert_="object",
    )

    # Dataloaders
    collate_fn = Collate(
        num_latents_per_step=cfg.model.num_latents,  # This was tied in train_poyo_1.py
        step=1.0 / 8,
        sequence_length=sequence_length,
        unit_vocab=model.unit_vocab,  # Unit vocab will be lazy loaded via the model.
        decoder_registry=decoder_registry,
        weight_registry=weight_registry,
    )

    loader_kwargs = dict(
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # For debugging. we allow the user to set num_workers to 0.
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    log.info(f"Training on {len(train_dataset)} samples")
    log.info(f"Training on {len(train_dataset.unit_names)} units")
    log.info(f"Training on {len(train_dataset.session_names)} sessions")

    # No need to explicitly use DDP with the model, lightning does this for us.
    max_lr = cfg.base_lr * cfg.batch_size

    if cfg.epochs > 0 and cfg.steps == 0:
        epochs = cfg.epochs
    elif cfg.steps > 0 and cfg.epochs == 0:
        epochs = cfg.steps // len(train_loader) + 1
    else:
        raise ValueError("Must specify either epochs or steps")

    print(f"Epochs: {epochs}")

    if cfg.finetune_epochs > 0:
        # if cfg.finetune_path is None:
        #    raise ValueError("Must specify finetune_path if finetune_epochs > 0")
        model.freeze_middle()

    # only parameters to tune are the unit embeddings and the task embedding
    optimizer = Lamb(
        model.parameters(),  # filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
        div_factor=1,
    )

    # Now we create the model wrapper. It's a simple shim that contains the train and
    # test code.
    wrapper = train_wrapper.TrainWrapper(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Check for the finetune path.
    if cfg.finetune_path is not None:
        if cfg.ckpt_path is not None:
            raise ValueError("Cannot specify both ckpt_path and finetune_path")

        # Load the checkpoint and partially apply it to the model.
        checkpoint = torch.load(cfg.finetune_path)
        state_dict = checkpoint["state_dict"]

        # Get the model's state_dict.
        model_state_dict = wrapper.state_dict()

        def compatible(a, b):
            if torch.is_tensor(a) and torch.is_tensor(b):
                return a.shape == b.shape
            elif a == b:
                return True

        # Filter the state_dict to only include the parameters we want to load.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict and compatible(v, model_state_dict[k])
        }
        # Load the state_dict.
        wrapper.load_state_dict(state_dict, strict=False)

        # Now list the parameters that haven't been loaded.
        missing_keys = [
            k for k in model_state_dict.keys() if k not in state_dict.keys()
        ]

        if len(missing_keys) > 0:
            print(
                "Missing keys not loaded from finetune_path: "
                + str(missing_keys)
            )

        # Re-bind the train loader to the vocabulary.
        train_loader.unit_vocab = model.unit_vocab

    tb = lightning.pytorch.loggers.tensorboard.TensorBoardLogger(
        save_dir=cfg.log_dir,
    )

    anonymous = hasattr(cfg, "anonymous") and cfg.anonymous
    wandb = lightning.pytorch.loggers.WandbLogger(
        name=cfg.name, project="poyo", log_model=True, anonymous=anonymous,
    )
    print(f"Wandb ID: {wandb.version}")

    callbacks = [
        ModelSummary(
            max_depth=2
        ),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            dirpath=f"logs/lightning_logs/{wandb.version}",
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        train_wrapper.CustomValidator(val_dataset, collate_fn),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
    ]

    if cfg.finetune_epochs > 0:
        callbacks.append(
            train_wrapper.MiddleFreezeUnfreeze(cfg.finetune_epochs)
        )

    trainer = lightning.Trainer(
        logger=[tb, wandb],
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=epochs,
        log_every_n_steps=1,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        num_sanity_val_steps=0,
        precision=cfg.precision,
        reload_dataloaders_every_n_epochs=5,
        accelerator="gpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes
    )

    log.info(f"Local rank/node rank/world size/num nodes: {trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/trainer.num_nodes")

    for logger in trainer.loggers:
        # OmegaConf.to_container converts the config object to a dictionary.
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # To resume from a checkpoint rather than training from scratch,
    # set ckpt_path on the command line.
    trainer.fit(wrapper, train_loader, [0], ckpt_path=cfg.ckpt_path)
    # [0] is a hack to force the validation callback to be called.


# This loads the config file using Hydra, similar to Flags, but composable.
@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train.yaml"
)
def main(cfg: DictConfig):
    # Train the whole thing.
    # This inner function is unnecessary, but I keep it here to maintain
    # a parallel to the original code.
    run_training(cfg)


if __name__ == "__main__":
    main()
