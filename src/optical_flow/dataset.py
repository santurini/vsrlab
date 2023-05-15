import os.path as osp
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path: Union[Path, str],
                 split: str = "train",
                 size: float = 0.9,
                 augmentation=None,
                 compression=None
                 ) -> None:

        self.root = "/home/aghinassi/Desktop/MergedVSR"
        self.path = list(sorted(Path(path).glob('*')))
        self.split = split
        self.augmentation = augmentation
        self.compression = compression

        split_point = int(len(self.path) * size)

        if split == 'train':
            self.path = self.path[:split_point]
        elif split == 'val':
            self.path = self.path[split_point:]

    def __len__(self) -> int:
        return len(self.path)

    def __getitem__(self, idx: int):
        supp, ref = self.get_path(self.path[idx])
        supp = to_tensor(Image.open(supp))
        ref = to_tensor(Image.open(ref))

        sequence = torch.stack([ref, supp])
        optical_flow = torch.load(self.path[idx])

        sequence, optical_flow = self.augmentation(sequence, optical_flow)
        sequence = self.compression(sequence)

        return (sequence[0], sequence[1]), optical_flow

    def get_path(self, path):
        path = str(path).split('_')
        video_name = '_'.join(path[:2])
        print(osp.join(self.root, video_name))
        print(list((self.root / video_name).glob(path[-2])))
        supp = list((self.root / video_name).glob(f"{path[-2]}.*"))[0]
        ref = list((self.root / video_name).glob(f"{path[-1].stem}.*"))[0]

        return supp, ref
