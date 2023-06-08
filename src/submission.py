import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import omegaconf
import torch
from torchvision.utils import save_image

from core import PROJECT_ROOT
from core.utils import (
    build_test_model,
    get_video
)

warnings.filterwarnings('ignore')

@torch.no_grad()
def run(config: omegaconf.DictConfig):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    cfg = omegaconf.OmegaConf.load(os.path.join(config.cfg_dir, "config.yaml"))
    ckpt_path = os.path.join(config.cfg_dir, "last.ckpt")
    model = build_test_model(cfg.train.model, device, ckpt_path)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    pool = ThreadPoolExecutor(config.num_workers)
    video_paths = list(Path(config.lr_dir).glob('*'))

    for i, video in enumerate(video_paths):
        video_name = os.path.basename(video)
        output_folder = os.path.join(config.out_dir, os.path.basename(video_name))

        model.eval()
        dt = time.time()

        print("Test Video {} / {}".format(i + 1, len(video_paths)))
        video_lr = get_video(video, pool)
        print('Loaded Video --> {}'.format(video_name))

        outputs = []
        windows = list(range(0, video_lr.size(1), config.window_size))
        for i in windows:
            lr = video_lr[:, i:i + config.window_size, ...].to(device, non_blocking=True)
            sr, _ = model(lr)
            outputs.append(sr)

        outputs = torch.cat(outputs, dim=1)

        print('Saving Video to --> {}'.format(output_folder))
        list(pool.map(
            lambda x: save_image(x[1], os.path.join(output_folder, "img{:05d}.png".format(x[0]))),
            enumerate(outputs[0]),
        ))

        dt = time.time() - dt
        print(f"Inference Time --> {dt:2f}\n")

    return config.out_dir

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
