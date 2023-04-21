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
from core.utils import get_model_state_dict, save_test_config, batched
from core.augmentations import read_video, write_video

import pandas as pd
from pathlib import Path
from piqa import PSNR, SSIM, MS_SSIM, LPIPS

pylogger = logging.getLogger(__name__)

psnr = PSNR().cuda()
ms_ssim = MS_SSIM().cuda()
ssim = SSIM().cuda()
lpips = LPIPS().cuda()

@torch.no_grad()
def test(cfg: DictConfig) -> str:
    output_path = save_test_config(cfg)
    pylogger.info(f"Output path <{output_path}>")

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module.model['_target_']}>")
    model: nn.Module = hydra.utils.instantiate(cfg.nn.module.model, _recursive_=False)

    pylogger.info(f"Loading pretrained weights: <{cfg.finetune}>")
    state_dict = get_model_state_dict(cfg.finetune)
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    model.eval()

    df_final = pd.DataFrame()
    for path in Path(cfg.path_lr).glob('*'):

        lr_frame_path = Path(output_path) / Path(path).stem
        lr_frame_path.mkdir(parents=True)

        pylogger.info(f"Reading LR video: <{path}>")
        lr_video, *_ = read_video(str(path))
        hr_video, c, r, h, w = read_video(os.path.join(cfg.path_hr, path.name))

        pylogger.info(f"<Processing video>")

        df = pd.DataFrame()
        i = 0
        for window_lr, window_hr in zip(batched(lr_video, cfg.window_size), batched(hr_video, cfg.window_size)):

            window_lr = torch.stack([to_tensor(frame.to_rgb().to_image()) for frame in window_lr]).cuda()

            out = model.test(window_lr.unsqueeze(0)).squeeze(0).clamp(0, 1)
            del window_lr

            window_hr = torch.stack([to_tensor(frame.to_rgb().to_image()) for frame in window_hr]).cuda()

            metrics = {
                "PSNR": psnr(out, window_hr),
                "MS-SSIM": ms_ssim(out, window_hr),
                "SSIM": ssim(out, window_hr),
                "LPIPS": lpips(out, window_hr),
            }

            for frame in out:
                file_name = lr_frame_path / f"frame_{i:04d}.png"
                to_pil_image(frame).save(file_name)
                i += 1

            del out

            df = df.append([metrics], ignore_index=True)


        pylogger.info(f"<Appending metrics>")
        metrics = df.mean(axis=0).to_dict()
        df_final = df_final.append([metrics], ignore_index=True)

    metrics_path = os.path.join(output_path, 'metrics.csv')
    pylogger.info(f"<Saving metrics csv to <{metrics_path}>")
    df_final.mean(axis=0).to_frame(name='Score').to_csv(metrics_path)

    return output_path

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test", version_base="1.3")
def main(config: omegaconf.DictConfig):
    test(config)

if __name__ == "__main__":
    main()
