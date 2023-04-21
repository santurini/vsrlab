import os
import warnings

warnings.filterwarnings("ignore")

import logging
from itertools import islice

import torch
import torch.nn.functional as F
import hydra
import omegaconf
from omegaconf import DictConfig

from core import PROJECT_ROOT
from core.utils import get_state_dict, save_test_config
from core.augmentations import read_video, write_video

import pandas as pd
from pathlib import Path
from piqa import PSNR, SSIM, MS_SSIM, LPIPS

pylogger = logging.getLogger(__name__)

@torch.no_grad()
def test(cfg: DictConfig) -> str:
    output_path = save_test_config(cfg)
    pylogger.info(f"Output path <{output_path}>")

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module.model['_target_']}>")
    model: nn.Module = hydra.utils.instantiate(cfg.nn.module.model, _recursive_=False)

    pylogger.info(f"Loading pretrained weights: <{cfg.finetune}>")
    state_dict = get_state_dict(cfg.finetune)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()

    df_final = pd.DataFrame()
    for path in Path(cfg.path_lr).glob('*'):
        pylogger.info(f"Reading video: <{path}>")
        lr_video, *_ = read_video(str(path), iterator=True)
        hr_video, c, r, h, w = read_video(os.path.join(cfg.path_hr, path.name), iterator=True)

        pylogger.info(f"Processing video>")
        df = pd.DataFrame()
        for window_lr, window_hr in zip(islice(lr_video, 0, cfg.window_size), islice(hr_video, 0, cfg.window_size)):

            window_lr = torch.stack([F.to_tensor(frame.to_image()) for frame in window_lr])
            window_hr = torch.stack([F.to_tensor(frame.to_image()) for frame in window_hr])
            out = model(window_lr.unsqueeze(0)).squeeze(0)
            del window_lr

            metrics = {
                "PSNR": PSNR(out, window_hr),
                "MS-SSIM": MS_SSIM(out, window_hr),
                "SSIM": SSIM(out, window_hr),
                "LPIPS": LPIPS(out, window_hr),
            }

            df = df.append([metrics], ignore_index=True)

        ## DA RIFARE
        video_path = os.path.join(output_path, path.name)
        write_video(video_path, out, codec=c, rate=r, crf=0, height=h, width=w)

        df_final = df_final.append([metrics], ignore_index=True)

    metrics_path = os.path.join(output_path, 'metrics.csv')
    df.mean(axis=0).to_frame(name='Score').to_csv(metrics_path)

    return output_path

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test", version_base="1.3")
def main(config: omegaconf.DictConfig):
    test(config)

if __name__ == "__main__":
    main()
