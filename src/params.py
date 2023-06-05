import os.path as osp
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mmcv
import omegaconf
import pandas as pd
import torch
from mmcv import build_from_cfg
from mmcv.runner import load_checkpoint
from mmedit.models.registry import MODELS

from core.utils import (
    build_test_model,
    get_video
)

warnings.filterwarnings('ignore')

C, H, W, WINDOW_SIZE, FPS, CRF = 3, 480, 640, 28, 6, 30

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def init_model(config, checkpoint=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    return model

@torch.no_grad()
def run():
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    for cfg_dir in ['/home/aghinassi/Desktop/checkpoints/basic_og']:
        # [x for x in Path('/home/aghinassi/Desktop/checkpoints').glob('*') if x.is_dir()]:

        # Encapsulate the model on the GPU assigned to the current process
        if osp.basename(cfg_dir) == "basic_og":
            cfg = osp.join(cfg_dir, "realbasicvsr_x4.py")
            ckpt_path = osp.join(cfg_dir, "RealBasicVSR_x4.pth")
            model = init_model(cfg, ckpt_path).to(device)
        else:
            cfg = omegaconf.OmegaConf.load(osp.join(cfg_dir, "config.yaml"))
            ckpt_path = osp.join(cfg_dir, "last.ckpt")
            model = build_test_model(cfg.train.model, device, ckpt_path)

        if osp.exists(f'/home/aghinassi/Desktop/time/{osp.basename(cfg_dir)}.csv'):
            print("Skipping model")
            continue

        results, tot_time = [], 0

        # Loop over the dataset multiple times
        print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
        pool = ThreadPoolExecutor(4)

        print('Testing Model --> {}'.format(osp.basename(cfg_dir)))
        print('Configuration: fps --> {} - crf -> {}\n'.format(FPS, CRF))
        video_folder = osp.join('/home/aghinassi/Desktop/compressed', f"fps={FPS}_crf={CRF}", "frames")
        video_paths = list(Path(video_folder).glob('*'))

        for i, video_lr_path in enumerate(video_paths):
            model.eval()

            print("Test Video {} / {}".format(i + 1, len(video_paths)))
            video_lr = get_video(video_lr_path, pool)

            windows = list(range(0, video_lr.size(1), WINDOW_SIZE))

            dt = time.time()
            for i in windows:
                lr = video_lr[:, i:i + WINDOW_SIZE, ...].to(device, non_blocking=True)
                _ = model(lr)

            dt = time.time() - dt
            print(f"Inference Time --> {dt:2f}\n")
            tot_time += dt

        results.append(
            {"model": osp.basename(cfg_dir), "avg_time": tot_time / len(video_paths), "params": get_params(model)}
        )
        print("Average Inference Time --> {}".format(tot_time / len(video_paths)))
        print("Number of Parameters --> {}".format(get_params(model)))

        results = pd.DataFrame(results)
        results.to_csv(f'/home/aghinassi/Desktop/time/{osp.basename(cfg_dir)}.csv', index=False)

    return results

def main():
    run()

if __name__ == "__main__":
    main()
