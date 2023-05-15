from pathlib import Path
from typing import Union

import torch
from PIL import Image
from einops import rearrange
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
        of_path = self.path[idx]
        supp, ref, optical_flow = self.get_path(of_path)
        supp = to_tensor(Image.open(supp))
        ref = to_tensor(Image.open(ref))

        sequence = torch.stack([ref, supp])

        sequence, optical_flow = self.augmentation(sequence, optical_flow)
        sequence, optical_flow = self.compression(sequence, optical_flow)

        return (sequence[0], sequence[1]), optical_flow

    def get_path(self, path):
        optical_flow = rearrange(
            torch.load(path, map_location="cpu"),
            '1 1 1 1 c h w -> c h w'
        )
        print(optical_flow.shape)
        path = str(path.stem).split('_')
        video_name = '_'.join(path[:2])
        supp = list((self.root / video_name).glob(f"{path[-2]}.*"))[0]
        ref = list((self.root / video_name).glob(f"{Path(path[-1]).stem}.*"))[0]

        return supp, ref, optical_flow
