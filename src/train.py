import warnings

warnings.filterwarnings("ignore")

import logging
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback

from core import PROJECT_ROOT
from core.utils import seed_index_everything, build_callbacks, get_state_dict, save_config

pylogger = logging.getLogger(__name__)

def run(cfg: DictConfig) -> str:
    save_config(cfg)
    seed_index_everything(cfg.train)

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False)

    if cfg.finetune:
        pylogger.info(f"Loading pretrained weights: <{cfg.finetune}>")
        state_dict = get_state_dict(cfg.finetune)
        model.load_state_dict(state_dict, strict=False)

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks)

    storage_dir: str = cfg.core.storage_dir
    logger: pl.loggers.Logger = hydra.utils.instantiate(cfg.train.logger)

    strategy = hydra.utils.instantiate(cfg.train.strategy)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.train.restore)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.save_dir

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
