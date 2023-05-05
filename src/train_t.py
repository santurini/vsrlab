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

def evaluate(model, device, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            lr, hr = data[0].to(device), data[1].to(device)
            sr, lq = model(lr)
            loss = loss_fn(sr, hr)

            #_, loss_dict = compute_loss(loss_fn, loss_dict, sr, hr)
            #metrics_dict = compute_metric(metric, metrics_dict, sr, hr)

    #logger.log_dict(loss_dict | metrics_dict, average_by=len(test_loader), stage="Val")
    #logger.log_images("Val", step, lr, sr, hr, lq)
    #save_checkpoint(cfg, model)
    return loss / len(test_loader)

def run(cfg: DictConfig):
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    if (local_rank == 0):
        print("world_size", world_size)

    #torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model_config = save_config(cfg)
    seed_index_everything(cfg.train)

    # Initialize logger
    #if rank == 0:
    #    print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
    #    logger = build_logger(cfg.train.logger)

    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)

    # Encapsulate the model on the GPU assigned to the current process
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    #if cfg.finetune is not None:
    #    model = restore_model(model, cfg.finetune, local_rank)

    # Prepare dataset and dataloader
    train_dl, val_dl, num_grad_acc, step, epoch = build_loaders(cfg)

    optimizer, scheduler = build_optimizer(cfg, model)

    loss_fn, loss_dict = CharbonnierLoss(), {"Loss": 0}
    metric, metrics_dict = build_metric(cfg.nn.module.metric)

    # Loop over the dataset multiple times
    print("Local Rank {} - Start Training ...".format(local_rank))
    while step < cfg.train.trainer.max_steps:
        dt = time.time()
        model.train()

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                sr, lq = model(lr)
                loss = loss_fn(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                #loss, loss_dict = compute_loss(loss_fn, loss_dict, sr, hr, lq)

            #step = update_weights(model, loss, scaler, scheduler, optimizer, num_grad_acc,
            #                       cfg.train.trainer.gradient_clip_val, step, i, len(train_dl))

            #metrics_dict = compute_metric(metric, metrics_dict, sr, hr)

        if local_rank == 0:
            #print("Logging on WandB ...")
            #logger.log_dict(loss_dict | metrics_dict, average_by=len(train_dl), stage="Train")
            #logger.log_images("Train", epoch, lr, sr, hr, lq)

            print("Rank {} -> Starting Evaluation ...".format(rank))
            evaluate(model, device, val_dl, loss_fn)

        dt = time.time() - dt
        print(f"Elapsed time epoch {epoch} --> {dt:2f}")
        epoch += 1

    #if rank == 0:
    #    logger.close()

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
