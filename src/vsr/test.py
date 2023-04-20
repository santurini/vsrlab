import warnings

warnings.filterwarnings("ignore")

import logging
from typing import List

import torch
import torch.nn as nn
import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig

from core import PROJECT_ROOT
from core.utils import seed_index_everything, build_callbacks, get_state_dict, save_config

pylogger = logging.getLogger(__name__)

def test(cfg: DictConfig) -> str:
    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module.model['_target_']}>")
    model: nn.Module = hydra.utils.instantiate(cfg.nn.module.model, _recursive_=False)

    pylogger.info(f"Loading pretrained weights: <{cfg.finetune}>")
    state_dict = get_state_dict(cfg.finetune)
    model.load_state_dict(state_dict, strict=False)



    return

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
