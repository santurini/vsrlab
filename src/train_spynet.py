import time
from typing import Sequence

import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from kornia.augmentation import Denormalize
from torch.utils.data import DataLoader

from core import PROJECT_ROOT
from core.utils import (
    build_loaders,
    build_optimizer,
    build_logger,
    save_checkpoint,
    save_config,
    cleanup
)
from optical_flow.models import spynet
from optical_flow.models.spynet.utils import (
    get_frames,
    get_flow,
    build_spynets,
    build_teacher,
    build_cleaner,
    update_weights,
    save_k_checkpoint
)

device = torch.device("cuda:{}".format(0))
denormalizer = Denormalize(mean=[.485, .406, .456], std=[.229, .225, .224], keepdim=True)

@torch.no_grad()
def evaluate(
        cfg,
        val_dl: DataLoader,
        criterion_fn: torch.nn.Module,
        Gk: torch.nn.Module,
        teacher: torch.nn.Module,
        cleaner: torch.nn.Module,
        prev_pyramid: torch.nn.Module = None,
        epoch: int = 0,
        k: int = -1,
        size: tuple = None,
        logger: nn.Module = None
):
    model.eval()
    val_loss = 0.0

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)
        x = get_frames(lr, cleaner, size)
        y, hr = get_flow(hr, teacher, size)

        if prev_pyramid is not None:
            with torch.no_grad():
                Vk_1 = prev_pyramid(x)
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Vk_1 = None

        predictions = Gk(x, Vk_1, upsample_optical_flow=False)

        if Vk_1 is not None:
            y = y - Vk_1

        loss = criterion_fn(y, predictions)

        val_loss += loss.detach().item()

    logger.log_dict({f"Loss {k}": val_loss / len(val_dl)}, epoch, "Val")
    logger.log_flow(f"Val {k}", epoch, hr, denormalizer(x[0]), predictions, y)
    save_k_checkpoint(cfg, k, current_level, logger, cfg.train.ddp)

def train_one_epoch(
        cfg,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer: nn.Module,
        scheduler: nn.Module,
        criterion_fn: torch.nn.Module,
        Gk: torch.nn.Module,
        teacher: torch.nn.Module,
        cleaner: torch.nn.Module,
        prev_pyramid: torch.nn.Module = None,
        epoch: int = 0,
        k: int = -1,
        size: tuple = None,
        logger: nn.Module = None
):
    Gk.train()
    dt = time.time()
    train_loss = 0.

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, data in enumerate(train_dl):
        lr, hr = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            x = get_frames(lr, cleaner, size)
            y, hr = get_flow(hr, teacher, size)

        if prev_pyramid is not None:
            with torch.no_grad():
                Vk_1 = prev_pyramid(x)
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            Vk_1 = None

        predictions = Gk(x, Vk_1, upsample_optical_flow=False)

        if Vk_1 is not None:
            y = y - Vk_1

        loss = criterion_fn(y, predictions)
        update_weights(loss, scheduler, optimizer)

        train_loss += loss.detach().item()

    logger.log_dict({f"Loss {k}": train_loss / len(train_dl)}, epoch, f"Train")
    print("x0", x[0].shape)
    print("denormalized", denormalizer(x[0]).shape)
    logger.log_flow(f"Train {k}", epoch, hr, denormalizer(x[0]), predictions, y)

    evaluate(
        cfg, val_dl, loss_fn, current_level, teacher,
        cleaner, trained_pyramid, epoch, k, size, logger
    )

    dt = time.time() - dt
    print(f"Epoch {epoch} Level {k} - Elapsed time --> {dt:2f}")


def train_one_level(cfg,
                    k: int,
                    previous: Sequence[spynet.BasicModule],
                    logger
                    ) -> spynet.BasicModule:
    print(f'Training level {k}...')

    train_dl, val_dl, _, _, epoch = build_loaders(cfg)

    current_level, trained_pyramid = build_spynets(cfg, k, previous, device)
    optimizer, scheduler = build_optimizer(current_level, cfg.train.optimizer, cfg.train.scheduler)
    teacher = build_teacher(cfg.train.teacher, device)
    cleaner = build_cleaner(cfg, device)

    loss_fn = spynet.nn.EPELoss()
    size = spynet.config.GConf(k).image_size

    for epoch in range(cfg.train.max_epochs):
        train_one_epoch(
            cfg,
            train_dl,
            val_dl,
            optimizer,
            scheduler,
            loss_fn,
            current_level,
            teacher,
            cleaner,
            trained_pyramid,
            epoch,
            k,
            size,
            logger
            )
    
    return current_level


def train(cfg):
    logger = build_logger(cfg)
    model_config = save_config(cfg)

    previous = []
    for k in range(cfg.train.k):
        previous.append(train_one_level(cfg, k, previous, logger))

    final = spynet.SpyNet(previous)
    save_checkpoint(cfg, final, logger, cfg.train.ddp)

    logger.close()

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    try:
        train(config)
    except Exception as e:
        if config.train.ddp:
            cleanup()
        wandb.finish()
        raise e

    if config.train.ddp:
        cleanup()

if __name__ == "__main__":
    main()
