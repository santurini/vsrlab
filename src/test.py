import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import omegaconf
import pandas as pd
import torch
from torchvision.utils import save_image

from core import PROJECT_ROOT
from core.utils import (
    build_test_model,
    build_metric,
    get_video,
    running_metrics
)

warnings.filterwarnings('ignore')

C, H, W = 3, 480, 640

@torch.no_grad()
def run(config: omegaconf.DictConfig):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    cfg = omegaconf.OmegaConf.load(os.path.join(config.cfg_dir, "config.yaml"))
    ckpt_path = os.path.join(config.cfg_dir, "last.ckpt")

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_test_model(cfg.train.model, device, ckpt_path)

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

                    sr, _ = model(lr)
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
    results.to_csv(os.path.join(output_folder, f'{os.path.basename(config.cfg_dir)}.csv'), index=False)

    return results

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
