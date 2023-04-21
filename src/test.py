import os
import av
import warnings

warnings.filterwarnings("ignore")

import logging

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import hydra
import omegaconf
from omegaconf import DictConfig

from core import PROJECT_ROOT
from core.utils import get_state_dict, save_test_config, batched
from core.augmentations import read_video, write_video

import pandas as pd
from pathlib import Path
from piqa import PSNR, SSIM, MS_SSIM, LPIPS

pylogger = logging.getLogger(__name__)

psnr = PSNR()
ms_ssim = MS_SSIM()
ssim = SSIM()
lpips = LPIPS()

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

        pylogger.info(f"Reading LR video: <{path}>")
        lr_video, *_ = read_video(str(path))
        hr_video, c, r, h, w = read_video(os.path.join(cfg.path_hr, path.name))

        pylogger.info(f"Processing video>")

        df = pd.DataFrame()
        out_video = []
        for window_lr, window_hr in zip(batched(lr_video, cfg.window_size), batched(hr_video, cfg.window_size)):

            pylogger.info(f"Loading LR window")
            window_lr = torch.stack([to_tensor(frame.to_image()) for frame in window_lr]).cuda()

            pylogger.info(f"Super resolve")
            out = model.test(window_lr.unsqueeze(0)).squeeze(0)

            del window_lr

            pylogger.info(f"Loading HR window")
            window_hr = torch.stack([to_tensor(frame.to_image()) for frame in window_hr]).cuda()

            metrics = {
                "PSNR": psnr(out, window_hr),
                "MS-SSIM": ms_ssim(out, window_hr),
                "SSIM": ssim(out, window_hr),
                "LPIPS": lpips(out, window_hr),
            }

            pylogger.info(f"Converting to Video Frame")
            out_video.extend([av.VideoFrame.from_image(to_pil_image(i)) for i in out])
            del out

            pylogger.info(f"Appending metrics")
            df = df.append([metrics], ignore_index=True)

        metrics = df.mean(axis=0).to_dict()
        df_final = df_final.append([metrics], ignore_index=True)

        video_path = os.path.join(output_path, path.name)
        write_video(video_path, out_video, codec=c, rate=r, crf=5, height=h, width=w)

    metrics_path = os.path.join(output_path, 'metrics.csv')
    df_final.mean(axis=0).to_frame(name='Score').to_csv(metrics_path)

    return output_path

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test", version_base="1.3")
def main(config: omegaconf.DictConfig):
    test(config)

if __name__ == "__main__":
    main()
