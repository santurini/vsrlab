import os.path as osp
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import omegaconf
import pandas as pd
import torch

from core.utils import (
    build_test_model,
    get_video
)

warnings.filterwarnings('ignore')

C, H, W, WINDOW_SIZE, FPS, CRF = 3, 480, 640, 32, 6, 30

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

    for cfg_dir in [x for x in Path('/home/aghinassi/Desktop/checkpoints').glob('*') if x.is_dir()]:

        # Encapsulate the model on the GPU assigned to the current process
        print('testing model --> {}'.format(osp.basename(cfg_dir)))
        if osp.basename(cfg_dir) == "basic_og":
            cfg = osp.join(cfg_dir, "realbasicvsr_x4.py")
            ckpt_path = osp.join(cfg_dir, "RealBasicVSR_x4.pth")
            model = init_model(cfg, ckpt_path).to(device)
        else:
            cfg = omegaconf.OmegaConf.load(osp.join(cfg_dir, "config.yaml"))
            ckpt_path = osp.join(cfg_dir, "last.ckpt")
            model = build_test_model(cfg.train.model, device, ckpt_path)

        results, tot_time = [], 0

        # Loop over the dataset multiple times
        print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
        pool = ThreadPoolExecutor(8)

        print('Configuration: fps --> {} - crf -> {}\n'.format(FPS, CRF))
        video_folder = osp.join('/home/aghinassi/Desktop/compressed', f"fps={FPS}_crf={CRF}", "frames")
        video_paths = list(Path(video_folder).glob('*'))

        for i, video_lr_path in enumerate(video_paths):
            model.eval()

            video_name = osp.basename(video_lr_path)
            video_hr_path = osp.join('/home/aghinassi/Desktop/groundtruth', f"fps={FPS}_crf=5", "frames", video_name)

            print("Test Video {} / {}".format(i + 1, len(video_paths)))
            video_hr, video_lr = get_video(video_hr_path, pool), get_video(video_lr_path, pool)
            print('Loaded Video --> {}'.format(video_name))

            windows = list(range(0, video_lr.size(1), WINDOW_SIZE))

            dt = time.time()
            for i in windows:
                lr, hr = video_lr[:, i:i + WINDOW_SIZE, ...].to(device, non_blocking=True), \
                    video_hr[:, i:i + WINDOW_SIZE, ...].to(device, non_blocking=True)

                sr, _ = model(lr)
                print(sr.size())

            dt = time.time() - dt
            print(f"Inference Time --> {dt:2f}\n")
            tot_time += dt

        results.append(
            {"model": osp.basename(cfg_dir), "avg_time": tot_time / len(video_paths), "params": get_params(model)}
        )

    results = pd.DataFrame(results)
    results.to_csv('/home/aghinassi/Desktop/time.csv', index=False)

    return results

def main():
    run()

if __name__ == "__main__":
    main()
