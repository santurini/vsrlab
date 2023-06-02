import time
import warnings
from typing import Sequence

import hydra
import omegaconf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from kornia.enhance import normalize
from torch.utils.data import DataLoader

from core import PROJECT_ROOT
from core.utils import (
    get_resources,
    build_logger,
    save_config,
    cleanup
)
from optical_flow.models import spynet
from optical_flow.models.spynet.utils import (
    clean_frames,
    load_data,
    setup_train,
    build_dl,
    build_cleaner,
    update_weights,
    save_k_checkpoint
)

warnings.filterwarnings('ignore')

@torch.no_grad()
def evaluate(
        cfg,
        val_dl: DataLoader,
        criterion_fn: torch.nn.Module,
        Gk: torch.nn.Module,
        cleaner: torch.nn.Module,
        optimizer: torch.nn.Module,
        scheduler: torch.nn.Module,
        prev_pyramid: torch.nn.Module = None,
        epoch: int = 0,
        k: int = -1,
        logger: nn.Module = None,
        device: str = 'cuda:0',
        rank: int = 0,
        world_size: int = 2
):
    Gk.eval()
    val_loss = 0.0

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, (x1, x2, y) in enumerate(val_dl):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        if i % 100 == 0:
            if rank == 0:
                print("Batch {}/{}".format(i, len(val_dl) - 1))

        with torch.cuda.amp.autocast():
            x1, x2 = clean_frames(cleaner, x1, x2)
            x = (
                normalize(x1, mean=torch.tensor([.485, .406, .456]), std=torch.tensor([.229, .225, .224])),
                normalize(x2, mean=torch.tensor([.485, .406, .456]), std=torch.tensor([.229, .225, .224]))
            )

            if prev_pyramid is not None:
                with torch.no_grad():
                    Vk_1 = prev_pyramid(x)
                    Vk_1 = F.interpolate(
                        Vk_1, scale_factor=2, mode='bilinear', align_corners=True)

            else:
                Vk_1 = torch.zeros_like(y)

            predictions = Gk(x, Vk_1, upsample_optical_flow=False) + Vk_1
            loss = criterion_fn(y, predictions)

            if cfg.train.ddp:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

            val_loss += loss.detach().item() / world_size

    if rank == 0:
        logger.log_dict({f"Loss {k}": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_flow(f"Val {k}", epoch, x1, predictions, y)
        save_k_checkpoint(cfg, k, Gk, optimizer, scheduler, epoch, logger, cfg.train.ddp)

def train_one_epoch(
        cfg,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer: nn.Module,
        scheduler: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        criterion_fn: torch.nn.Module,
        Gk: torch.nn.Module,
        cleaner: torch.nn.Module,
        prev_pyramid: torch.nn.Module = None,
        epoch: int = 0,
        k: int = -1,
        logger: nn.Module = None,
        device: str = 'cuda:0',
        rank: int = 0,
        world_size: int = 2
):
    Gk.train()
    dt = time.time()
    train_loss = 0.

    if prev_pyramid is not None:
        prev_pyramid.eval()

    for i, (x1, x2, y) in enumerate(train_dl):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        if i % 100 == 0:
            if rank == 0:
                print("Batch {}/{}".format(i, len(train_dl) - 1))

        with torch.cuda.amp.autocast():
            x1, x2 = clean_frames(cleaner, x1, x2)
            x = (
                normalize(x1, mean=torch.tensor([.485, .406, .456]), std=torch.tensor([.229, .225, .224])),
                normalize(x2, mean=torch.tensor([.485, .406, .456]), std=torch.tensor([.229, .225, .224]))
            )

            if prev_pyramid is not None:
                with torch.no_grad():
                    Vk_1 = prev_pyramid(x)
                    Vk_1 = F.interpolate(
                        Vk_1, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                Vk_1 = torch.zeros_like(y)

            predictions = Gk(x, Vk_1, upsample_optical_flow=False) + Vk_1
            loss = criterion_fn(y, predictions)

        update_weights(loss, Gk, scheduler, optimizer, scaler)
        train_loss += loss.detach().item()

    if rank == 0:
        logger.log_dict({f"Loss {k}": train_loss / len(train_dl)}, epoch, f"Train")
        logger.log_flow(f"Train {k}", epoch, x1, predictions, y)

    print("Starting Evaluation ...")

    evaluate(
        cfg, val_dl, criterion_fn, Gk, cleaner, optimizer, scheduler,
        prev_pyramid, epoch, k, logger, device, rank, world_size
    )

    if rank == 0:
        dt = time.time() - dt
        print(f"Epoch {epoch} Level {k} - Elapsed time --> {dt:2f}")


def train_one_level(cfg,
                    k: int,
                    previous: Sequence[spynet.BasicModule],
                    scaler,
                    logger,
                    device,
                    rank,
                    local_rank,
                    world_size
                    ) -> spynet.BasicModule:
    if rank == 0: print(f'Training level {k}...')

    if rank == 0: print("Preparing datasets")
    train_ds, val_ds = load_data(cfg, k)

    if rank == 0: print("Preparing dataloaders")
    train_dl, val_dl, epoch = build_dl(train_ds, val_ds, cfg)

    if rank == 0: print("Instantiating pyramids")
    current_level, trained_pyramid, optimizer, scheduler, start_epoch = setup_train(cfg, k, previous,
                                                                                    cfg.train.optimizer,
                                                                                    cfg.train.scheduler, device,
                                                                                    local_rank)

    if rank == 0: print("Instantiating cleaner")
    cleaner = build_cleaner(cfg, device)

    loss_fn = nn.L1Loss()
    max_epochs = cfg.train.max_epochs[k]

    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(
            cfg,
            train_dl,
            val_dl,
            optimizer,
            scheduler,
            scaler,
            loss_fn,
            current_level,
            cleaner,
            trained_pyramid,
            epoch,
            k,
            logger,
            device,
            rank,
            world_size
        )
    
    return current_level

def train(cfg):
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    # Initialize logger
    if rank == 0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        resume = False if cfg.train.restore is None else True
        logger = build_logger(cfg, resume)
        model_config = save_config(cfg)
    else:
        logger = None

    device = torch.device("cuda:{}".format(local_rank))
    scaler = torch.cuda.amp.GradScaler()

    previous = []
    for k in range(cfg.train.start_k, cfg.train.k):
        previous.append(
            train_one_level(cfg, k, previous, scaler, logger,
                            device, rank, local_rank, world_size)
        )

    final = spynet.SpyNet(previous)
    model_state_dict = final.module.state_dict() if cfg.train.ddp else final.state_dict()

    base_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id
    )

    save_path = os.path.join(
        base_path,
        "checkpoint_last.tar"
    )

    if rank == 0:
        torch.save(model_state_dict, save_path)
        logger.save(save_path, base_path)
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
