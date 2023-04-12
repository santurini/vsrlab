import logging
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import os
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import seed_everything, Callback

import torch
from torch.nn import Sequential

from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_model_state_file,
    get_optim_files,
)

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

def convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir: Union[str, Path],
        output_file: Union[str, Path],
        tag: Union[str, None] = None
) -> Dict[str, Any]:

    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    deepspeed_states = [
        "module",
        "optimizer",
        "lr_scheduler",
        "csr_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "dp_world_size",
        "mp_world_size",
    ]
    checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    model_file = get_model_state_file(checkpoint_dir, zero_stage)
    client_state = torch.load(model_file, map_location=CPU_DEVICE)
    client_state = {key: value for key, value in client_state.items() if key not in deepspeed_states}
    state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
    client_state["state_dict"] = state_dict

    if not os.path.exists(os.path.abspath(output_file)):
        print(f"Saving fp32 state dict to {output_file}")
        torch.save(client_state, output_file)

    return client_state

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

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
