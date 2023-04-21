import logging
import os
import yaml
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import hydra
import numpy as np
from collections import OrderedDict
import torch

from omegaconf import DictConfig, ListConfig, OmegaConf
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

def save_config(cfg):
    save_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id,
        "config.yaml"
    )
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    with open(save_path, 'w') as file:
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        file.write(yaml_str)

def save_test_config(cfg):
    model = cfg.model_name
    version = cfg.finetune.stem
    output_path = ''.join(['sr', Path(cfg.path_lr).name.partition('lr')[-1]])

    save_path = '_'.join([
        output_path,
        model,
        version
        ])

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    with open(f"{save_path}.yaml", 'w') as file:
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        file.write(yaml_str)

    return save_path

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