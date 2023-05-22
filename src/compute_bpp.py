import os
import time
from pathlib import Path

import hydra
import omegaconf
import pandas as pd

from core import PROJECT_ROOT

C, H, W = 3, 480, 640

def run(config):
    rank, local_rank, world_size = (0, 0, 1)
    video_pd = []

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))

    for fps in [15, 12, 10, 8, 6]:
        for crf in [30, 32, 34, 36, 38, 40]:
            print('Configuration: fps -> {} - crf -> {} '.format(fps, crf))
            video_folder = os.path.join(config.lr_dir, f"fps={fps}_crf={crf}", "frames")
            output_folder = os.path.join(config.out_dir, os.path.basename(config.cfg_dir))
            video_paths = list(Path(video_folder).glob('*'))
            bpp = 0

            for video_lr_path in video_paths:
                dt = time.time()

                video_name = os.path.basename(video_lr_path)
                video_hr_path = os.path.join(config.hr_dir, f"fps={fps}_crf=5", "frames", video_name)
                save_folder = os.path.join(output_folder, f"fps={fps}_crf={crf}", video_name)
                Path(save_folder).mkdir(exist_ok=True, parents=True)

                n_frames = len(list(Path(video_lr_path).glob('*')))
                # size_bits = (Path(config.lr_dir) / f"fps={fps}_crf=5" / "video" / video_name).stat().st_size * 8
                size_bits = (Path(config.lr_dir) / f"fps={fps}_crf={crf}" / "video" / video_name).stat().st_size * 8
                bpp += size_bits / (C * H * W * n_frames)

            video_pd.append(
                {"bpp": bpp / len(video_paths), "fps": fps, "crf": crf})

            dt = time.time() - dt
            print(f"Inference Time --> {dt:2f}")

    pd.DataFrame(video_pd).to_csv('/mnt/hdd/dataset/pexels/compressed/bpp_compressed.csv')

    return pd.DataFrame(video_pd)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
