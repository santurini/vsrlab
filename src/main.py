import warnings

warnings.filterwarnings("ignore")

from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from core import PROJECT_ROOT
from core.utils import seed_index_everything, build_callbacks

def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg.train)

    # Instantiate datamodule
    print(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    # Instantiate model
    print(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False)
    model.load_from_checkpoint('/home/aghinassi/Desktop/nn-lab/storage/video-super-resolution/ltkflmx0/checkpoints/best.ckpt/checkpoint/zero_pp_rank_1_mp_rank_00_model_states.pt')

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks)

    storage_dir: str = cfg.core.storage_dir
    logger: pl.loggers.Logger = hydra.utils.instantiate(cfg.train.logger)

    try:
        strategy = hydra.utils.instantiate(cfg.train.strategy)
    except:
        strategy = cfg.train.strategy

    print("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        **cfg.train.trainer,
    )

    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.train.restore.ckpt_path)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()
    print(callbacks[1].best_model_path)
    return logger.save_dir

@hydra.main(config_path=str(PROJECT_ROOT / "conf_test"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    run(cfg.cfg)

if __name__ == "__main__":
    main()
