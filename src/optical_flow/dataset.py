from pathlib import Path
from random import shuffle
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

        self.root = Path("/home/aghinassi/Desktop/MergedVSR")
        self.path = shuffle(list(Path(path).glob('*')))[:10000]
        self.split = split
        self.augmentation = augmentation
        self.compression = compression

        split_point = int(len(self.path) * size)
        print("Total Size: {} -> Train Size: {}".format(len(self.path), int(len(self.path) * size)))

        if split == 'train':
            self.path = self.path[:split_point]
        elif split == 'val':
            self.path = self.path[split_point:]

    def __len__(self) -> int:
        return len(self.path)

    def __getitem__(self, idx: int):
        frame1, frame2, optical_flow = self.get_path(self.path[idx])
        frame1 = to_tensor(Image.open(frame1))
        frame2 = to_tensor(Image.open(frame2))

        sequence = torch.stack([frame1, frame2])

        sequence, optical_flow = self.augmentation(sequence, optical_flow)
        sequence, optical_flow = self.compression(sequence, optical_flow)

        return sequence[0], sequence[1], optical_flow

    def get_path(self, path):
        optical_flow = torch.load(path, map_location="cpu")
        path = str(path.stem).split('_')
        video_name = '_'.join(path[:2])
        frame1 = list((self.root / video_name).glob(f"{path[-2]}.*"))[0]
        frame2 = list((self.root / video_name).glob(f"{Path(path[-1]).stem}.*"))[0]

        return frame1, frame2, optical_flow
