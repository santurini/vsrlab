import logging
from typing import Optional, Sequence

import hydra
import omegaconf
import pytorch_lightning as pl
from core import PROJECT_ROOT
from omegaconf import DictConfig
from torch.utils.data import DataLoader

pylogger = logging.getLogger(__name__)

class DataModuleVSR(pl.LightningDataModule):
    def __init__(
            self,
            datasets: DictConfig,
            num_workers: int,
            batch_size: int,
            prefetch_factor: int
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: Optional[str] = None):
        self.train_ds = hydra.utils.instantiate(self.datasets.train, _recursive_=False)
        self.val_ds = hydra.utils.instantiate(self.datasets.val, _recursive_=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

if __name__ == "__main__":
    main()
