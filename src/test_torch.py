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

@torch.no_grad()
def run(cfg: DictConfig):
    model_config = save_config(cfg)
    seed_index_everything(cfg.train)

    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    model = restore_model(model, cfg.finetune, local_rank)

    print('build metrics and losses ...')
    metric  = build_metric(cfg.nn.module.metric).to(device)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    for video in video_paths:
        model.eval(); dt = time.time()
        video_metrics = {k: 0 for k in cfg.nn.module.metric.metrics}

        video_seq = get_video(video)

        for i, seq in enumerate(video_seq):
            lr, hr = seq[0].to(device), seq[1].to(device)

            sr, _ = model(lr)
            video_metrics = running_metrics(video_metrics, metric, sr, hr)

        logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
        logger.log_dict({k: v / len(train_dl) for k, v in train_metrics.items()}, epoch, "Train")
        logger.log_images("Train", epoch, lr, sr, hr, lq)

        dt = time.time() - dt
        print(f"Inference Time --> {dt:2f}")

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
