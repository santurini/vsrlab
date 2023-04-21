import os
import warnings

warnings.filterwarnings("ignore")

import logging

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

def test(cfg: DictConfig) -> str:
    os.mkdir(cfg.output_path)
    output_path = save_test_config(cfg)

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module.model['_target_']}>")
    model: nn.Module = hydra.utils.instantiate(cfg.nn.module.model, _recursive_=False)

    pylogger.info(f"Loading pretrained weights: <{cfg.finetune}>")
    state_dict = get_state_dict(cfg.finetune)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    df = pd.DataFrame()
    for path in Path(cfg.path_lr).glob('*'):
        lr_video, *_ = read_video(str(path))
        hr_video, c, r, h, w = read_video(cfg.path_hr + path.name)
        out = model(lr_video.unsqueeze(0)).squeeze(0)

        video_path = os.path.join(output_path, path.name)
        write_video(video_path, out, codec=c, rate=r, crf=0, height=h, width=w)

        metrics = {
            "PSNR": PSNR(out, hr_video),
            "MS-SSIM": MS_SSIM(out, hr_video),
            "SSIM": SSIM(out, hr_video),
            "LPIPS": LPIPS(out, hr_video),
        }

        df = df.append([metrics], ignore_index=True)

    metrics_path = os.path.join(output_path, 'metrics.csv')
    df.mean(axis=0).to_frame(name='Score').to_csv(metrics_path)

    return output_path

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="test", version_base="1.3")
def main(config: omegaconf.DictConfig):
    test(config)

if __name__ == "__main__":
    main()
