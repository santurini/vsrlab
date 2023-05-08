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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script for VSR')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('cfg_dir', help='directory of model config')
    parser.add_argument('lr_dir', help='directory of the input lr video')
    parser.add_argument('hr_dir', help='directory of the input hr video')
    parser.add_argument('out_dir', help='directory of the output video')
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='maximum sequence length to be processed')
    args = parser.parse_args()
    return args


@torch.no_grad()
def run(cfg, args):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    cfg = OmegaConf.load(os.path.join(args.cfg_dir, "config.yaml"))
    ckpt_path = os.path.join(args.cfg_dir, "last.ckpt")

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_model(cfg.nn.module.model, device, local_rank, False)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    model = restore_model(model, ckpt_path, local_rank)

    print('build metrics and losses ...')
    metric, video_metrics  = build_metric(cfg.metric).to(device), \
                                {k: 0 for k in cfg.metric.metrics}

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    video_paths = list(Path(args.lr_dir).glob('*'))

    for video_lr_path in video_paths:
        model.eval(); dt = time.time()

        video_name = os.path.basename(video_lr_path)
        video_hr_path = os.path.join(args.hr_dir, video_name)


        video_hr, video_lr = get_video(video_hr_path).to(device), \
                                get_video(video_lr_path).to(device)

        outputs = []
        for i in range(0, video_lr.size(1), args.window_size):
            lr, hr = video_lr[:, i:i + args.window_size, ...].to(device), \
                        video_hr[:, i:i + args.window_size, ...].to(device)
            sr = model(lr)
            outputs.append(sr.cpu())

        outputs = torch.cat(outputs, dim=1)[0]
        for i, img in enumerate(outputs):
            save_image(img, os.path.join(args.out_dir, video_name, f"img{i}.png"))

        video_metrics = running_metrics(video_metrics, metric, outputs, video_hr)

        dt = time.time() - dt
        print(f"Inference Time --> {dt:2f}")

    video_metrics = {k: v / len(video_paths) for k,v in video_metrics.items()}

    return args.out_dir

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
