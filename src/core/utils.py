import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import hydra
import numpy as np
from collections import OrderedDict
import torch

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import seed_everything, Callback
from torch.nn import Sequential

CPU_DEVICE = torch.device("cpu")

pylogger = logging.getLogger(__name__)

def seed_index_everything(train_cfg: DictConfig, sampling_seed: int = 42) -> Optional[int]:
    if "seed_index" in train_cfg and train_cfg.seed_index is not None:
        seed_index = train_cfg.seed_index
        np.random.seed(sampling_seed)
        seeds = np.random.randint(np.iinfo(np.int32).max, size=max(42, seed_index + 1))
        seed = seeds[seed_index]
        seed_everything(seed)
        pylogger.info(f"Setting seed {seed} from seeds[{seed_index}]")
        return seed
    else:
        pylogger.warning("The seed has not been set! The reproducibility is not guaranteed.")
        return None

def get_state_dict(path):
    return torch.load(path)['state_dict']

def build_callbacks(cfg: ListConfig) -> List[Callback]:
    callbacks = list()
    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks

def build_transform(cfg: ListConfig) -> List[Sequential]:
    augmentation = list()
    for aug in cfg:
        pylogger.info(f"Adding augmentation <{aug['_target_'].split('.')[-1]}>")
        augmentation.append(hydra.utils.instantiate(aug, _recursive_=False))
    return Sequential(*augmentation)

def ds_checkpoint_dir(
        checkpoint_dir: Union[str, Path],
        tag: Union[str, None] = None
) -> str:
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path) as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    directory = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")
    return directory

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
