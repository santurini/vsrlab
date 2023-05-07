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

def evaluate(rank, world_size, epoch, model, logger, device, val_dl, loss_fn, of_loss_fn, metric, cfg):
    model.eval()
    val_loss = 0
    val_metrics = {k: 0 for k in cfg.nn.module.metric.metrics}
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            lr, hr = data[0].to(device), data[1].to(device)
            sr, lq = model(lr)
            loss = compute_loss(loss_fn, sr, hr, lq, of_loss_fn)

            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            val_loss += loss.detach().item() / world_size
            val_metrics = running_metrics(val_metrics, metric, sr, hr)

        if rank == 0:
            logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
            logger.log_dict({k: v / len(val_dl) for k,v in val_metrics}, epoch, "Val")
            logger.log_images("Val", epoch, lr, sr, hr, lq)
            save_checkpoint(cfg, model)


def run(cfg: DictConfig):
    model_config = save_config(cfg)
    seed_index_everything(cfg.train)

    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    # Initialize logger
    if rank==0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        logger = build_logger(cfg.train.logger)
    else:
        logger = None

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    # Mixed precision
    print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if cfg.finetune is not None:
        model = restore_model(model, cfg.finetune, local_rank)

    # Prepare dataset and dataloader
    print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    print('build optimizer and scheduler ...')
    optimizer, scheduler = build_optimizer(cfg, model)

    print('build metrics and losses ...')
    of_loss_fn = OpticalFlowConsistency(device, 0.01) if cfg.train.use_of_loss else None
    loss_fn, train_loss = CharbonnierLoss(), 0
    metric, train_metrics = build_metric(cfg.nn.module.metric).to(device), {k: 0 for k in cfg.nn.module.metric.metrics}

    # Loop over the dataset multiple times
    print("Local Rank {} - Start Training ...".format(local_rank))
    for epoch in range(cfg.train.trainer.max_epochs):
        dt = time.time()
        model.train()

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                sr, lq = model(lr)
                loss = compute_loss(loss_fn, sr, hr, lq, of_loss_fn)


            update_weights(model, loss, scaler, scheduler,
                            optimizer, num_grad_acc, gradient_clip_val, i)

            train_loss += loss.detach.item()
            train_metrics = running_metrics(metrics, metric, sr, hr)

        if rank == 0:
            print("Logging on WandB ...")
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_dict({k: v / len(train_dl) for k, v in train_metrics}, epoch, "Train")
            logger.log_images("Train", epoch, lr, sr, hr, lq)

        print("Starting Evaluation ...")
        evaluate(rank, world_size, epoch, model, logger, device,
                    val_dl, loss_fn, of_loss_fn, metric, cfg)

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
