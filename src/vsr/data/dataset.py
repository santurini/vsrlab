from pathlib import Path
from random import randint
from typing import Optional

import hydra
import omegaconf
import torch
from PIL import Image
from core import PROJECT_ROOT
from core.utils import build_transform
from kornia.geometry.transform import resize
from omegaconf import ListConfig
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

class DatasetSR(Dataset):
    def __init__(self,
                 path: str,
                 split: str,
                 size: int,
                 scale: Optional[int],
                 hr_augmentation: ListConfig = None,
                 lr_augmentation: ListConfig = None
                 ):
        super().__init__()
        self.path = list(Path(path).glob('*'))
        self.scale = scale
        self.split = split
        self.hr_augmentation = build_transform(hr_augmentation) if hr_augmentation else None
        self.lr_augmentation = build_transform(lr_augmentation) if lr_augmentation else None

        split_point = int(len(self.path) * size)

        if split == 'train':
            self.path = self.path[:split_point]
        elif split == 'val':
            self.path = self.path[split_point:]

    def __len__(self) -> int:
        return len(self.path)

    def __getitem__(self, index: int):
        hr_batch = self.load_img(self.path[index])

        if self.hr_augmentation:
            hr_batch = self.hr_augmentation(hr_batch)

        if self.lr_augmentation:
            lr_batch = self.lr_augmentation(hr_batch)
        else:
            h, w = hr_batch.shape[-2:]
            lr_batch = resize(hr_batch, (h // self.scale, w // self.scale))

        return lr_batch, hr_batch

    def __repr__(self) -> str:
        return f"Dataset({self.split=}, n_instances={len(self)})"

    @staticmethod
    def load_img(path):
        return to_tensor(Image.open(path))

class DatasetVSR(DatasetSR):
    def __init__(self,
                 seq: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = seq

    def __getitem__(self, index: int):
        hr_video = list(sorted(x for x in self.path[index].glob('*') if x.is_file()))
        hr_video = self.get_frames(hr_video, randint(0, len(hr_video) - self.seq))

        if self.hr_augmentation:
            hr_video = self.hr_augmentation(hr_video)

        if self.lr_augmentation:
            lr_video = self.lr_augmentation(hr_video)
        else:
            h, w = hr_video.shape[-2:]
            lr_video = resize(hr_video, (h // self.scale, w // self.scale))

        return lr_video, hr_video

    def get_frames(self, video, rnd):
        video = [self.load_img(i) for i in video[rnd:rnd + self.seq]]
        return torch.stack(video)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, _recursive_=False)

if __name__ == "__main__":
    main()