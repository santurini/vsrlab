import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid
from kornia.geometry.transform import resize

import os
import random
import numpy as np
import time
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.utils import clip_grad_norm_

import hydra
import omegaconf
from omegaconf import DictConfig
import wandb

from core.utils import *
from core import PROJECT_ROOT
from core.losses import CharbonnierLoss

import warnings

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

def evaluate(rank, model, logger, device, val_dl, step, loss_fn, metric, cfg):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            print("Batch {}/{}".format(i, len(val_dl)))
            lr, hr = data[0].to(device), data[1].to(device)
            sr, lq = model(lr)
            loss = compute_loss(loss_fn, sr, hr, lq)

            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

            if rank==0:
                logger.log_dict({"Loss": loss.detach().item() / world_size}, "Val")
                logger.log_dict(compute_metric(metric, sr, hr), "Val")

        if rank == 0:
            print("Logging on WandB ...")
            logger.log_images("Val", step, lr, sr, hr, lq)
            save_checkpoint(cfg, model)


def run(cfg: DictConfig):
    model_config = save_config(cfg)
    seed_index_everything(cfg.train)

    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    # Initialize logger
    #if rank == 0:
    pylogger.info("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
    logger = build_logger(cfg.train.logger)

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('model')
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    # Mixed precision
    print('scaler')
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if cfg.finetune is not None:
        model = restore_model(model, cfg.finetune, local_rank)

    # Prepare dataset and dataloader
    print('loaders')
    train_dl, val_dl, num_grad_acc, step, epoch = build_loaders(cfg)

    print('optimizer')
    optimizer, scheduler = build_optimizer(cfg, model)

    print('metrics')
    loss_fn = CharbonnierLoss()
    metric = build_metric(cfg.nn.module.metric).to(device)

    # Loop over the dataset multiple times
    pylogger.info("Local Rank {} - Start Training ...".format(local_rank))
    while step < cfg.train.trainer.max_steps:
        dt = time.time()
        model.train()

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)
            print("Batch {}/{}".format(i, len(train_dl)))

            with torch.cuda.amp.autocast():
                sr, lq = model(lr)
                loss = compute_loss(loss_fn, sr, hr, lq)

            loss = loss / num_grad_acc
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.train.trainer.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if rank==0:
                logger.log_dict({"Loss": loss.detach().item()}, "Train")
                logger.log_dict(compute_metric(metric, sr, hr), "Train")

        if rank == 0:
            print("Logging on WandB ...")
            logger.log_images("Train", step, lr, sr, hr, lq)

        print("Starting Evaluation ...")
        evaluate(rank, model, logger, device, val_dl,
                    step, loss_fn, metric, cfg)

        dt = time.time() - dt
        print(f"Elapsed time --> {dt:2f}")

    if rank == 0:
        logger.close()

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
