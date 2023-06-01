import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import mmcv
import omegaconf
import pandas as pd
import torch
from mmcv import build_from_cfg
from mmcv.runner import load_checkpoint
from mmedit.models.registry import MODELS
from torchvision.utils import save_image

from core import PROJECT_ROOT
from core.utils import (
    build_metric,
    get_video,
    running_metrics
)

warnings.filterwarnings('ignore')

C, H, W = 3, 480, 640

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
def run(config: omegaconf.DictConfig):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    cfg = os.path.join(config.cfg_dir, "configs/realbasicvsr_x4.py")
    ckpt_path = os.path.join(config.cfg_dir, "checkpoints/RealBasicVSR_x4.pth")

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = init_model(cfg, ckpt_path).to(device)

    print('build metrics and losses ...')
    metric, video_pd = build_metric(config.metric).to(device), []

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    pool = ThreadPoolExecutor(config.num_workers)

    for fps in [6, 8, 10]:
        for crf in [30, 32, 34]:
            print('Configuration: fps -> {} - crf -> {} '.format(fps, crf))
            video_folder = os.path.join(config.lr_dir, f"fps={fps}_crf={crf}", "frames")
            output_folder = os.path.join(config.out_dir, os.path.basename(config.cfg_dir))
            video_paths = list(Path(video_folder).glob('*'))
            metrics, bpp, cf = {k: 0 for k in config.metric.metrics}, 0, 0

            for video_lr_path in video_paths:
                model.eval()
                dt = time.time()

                video_name = os.path.basename(video_lr_path)
                video_hr_path = os.path.join(config.hr_dir, f"fps={fps}_crf=5", "frames", video_name)
                save_folder = os.path.join(output_folder, f"fps={fps}_crf={crf}", video_name)
                Path(save_folder).mkdir(exist_ok=True, parents=True)

                video_hr, video_lr = get_video(video_hr_path, pool), get_video(video_lr_path, pool)

                F = video_hr.size(1)
                size_bits_orig = (Path(config.hr_dir) / f"fps={fps}_crf=5" / "video" / video_name).stat().st_size * 8
                size_bits_comp = (Path(
                    config.lr_dir) / f"fps={fps}_crf={crf}" / "video" / video_name).stat().st_size * 8

                cf += size_bits_comp / size_bits_orig
                bpp += size_bits_comp / (C * H * W * F)

                outputs = []
                windows = list(range(0, video_lr.size(1), config.window_size))
                video_metrics, norm_factor = {k: 0 for k in config.metric.metrics}, len(windows)
                for i in windows:
                    lr, hr = video_lr[:, i:i + config.window_size, ...].to(device, non_blocking=True), \
                        video_hr[:, i:i + config.window_size, ...].to(device, non_blocking=True)

                    sr = model(lr, test_mode=True)['output']
                    outputs.append(sr)

                    video_metrics = running_metrics(video_metrics, metric, sr, hr)

                outputs = torch.cat(outputs, dim=1)

                list(pool.map(
                    lambda x: save_image(x[1], os.path.join(save_folder, "img{:05d}.png".format(x[0]))),
                    enumerate(outputs[0]),
                ))

                video_metrics = {k: v / norm_factor for k, v in video_metrics.items()}
                metrics = {k: (metrics[k] + video_metrics[k]) for k in set(metrics) & set(video_metrics)}

                dt = time.time() - dt
                print(f"Inference Time --> {dt:2f}")

            video_pd.append(
                {"cf": cf / len(video_paths), "bpp": bpp / len(video_paths), "fps": fps, "crf": crf} |
                {k: v / len(video_paths) for k, v in metrics.items()}
            )

    results = pd.DataFrame(video_pd)
    results.to_csv(os.path.join(output_folder, f'{os.path.basename(config.cfg_dir)}.csv'))

    return results

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()