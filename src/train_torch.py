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

import hydra
import omegaconf
from omegaconf import DictConfig
import wandb

from core.utils import *
from core import PROJECT_ROOT
from core.losses import CharbonnierLoss

import warnings

warnings.filterwarnings('ignore')

@torch.no_grad()
def evaluate(model, logger, device, test_loader, step, loss_fn, loss_dict, metric, metrics_dict, cfg):
    model.eval()
    for i, data in enumerate(test_loader):
        print('Batch: {}/{}'.format(i, len(test_loader)))
        lr, hr = data[0].to(device), data[1].to(device)
        sr, lq = model(lr)
        _, loss_dict = compute_loss(loss_fn, loss_dict, sr, hr)
        metrics_dict = compute_metric(metric, metrics_dict, sr, hr)

    logger.log_dict(loss_dict | metrics_dict, average_by=len(test_loader), stage="Val")
    logger.log_images("Val", step, lr, sr, hr, lq)
    save_checkpoint(cfg, model)


def run(cfg: DictConfig):
    model_config = save_config(cfg)
    seed_index_everything(cfg.train)

    rank, local_rank = get_resources() if cfg.train.ddp else (0, 0)

    # Initialize logger
    if local_rank == 0:
        logger = build_logger(cfg.train.logger)

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if cfg.finetune is not None:
        model = restore_model(model, cfg.finetune, local_rank)

    # Prepare dataset and dataloader
    train_dl, val_dl, num_grad_acc, step, epoch = build_loaders(cfg)

    optimizer, scheduler = build_optimizer(cfg, model)

    loss_fn, loss_dict = CharbonnierLoss(), {"Loss": 0}
    metric, metrics_dict = build_metric(cfg.nn.module.metric)

    # Loop over the dataset multiple times
    print("Start Training ...")
    while step < cfg.train.trainer.max_steps:
        dt = time.time()
        model.train()

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                sr, lq = model(lr)
                loss, loss_dict = compute_loss(loss_fn, loss_dict, sr, hr, lq)

            step = update_weights(model, loss, scaler, scheduler, optimizer, num_grad_acc,
                                   cfg.train.trainer.gradient_clip_val, step, i, len(train_dl))

            metrics_dict = compute_metric(metric, metrics_dict, sr, hr)

        if rank == 0:
            print("Logging on WandB ...")
            logger.log_dict(loss_dict | metrics_dict, average_by=len(train_dl), stage="Train")
            logger.log_images("Train", epoch, lr, sr, hr, lq)

            print("Starting Evaluation ...")
            evaluate(model, logger, device, val_dl, step,
                     loss_fn, loss_dict, metric, metrics_dict, cfg)

            dt = time.time() - dt
            print(f"Elapsed time epoch {epoch} --> {dt:2f}")

    if rank == 0:
        logger.close()

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
