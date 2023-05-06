from typing import Any, List
import wandb

import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from kornia.geometry.transform import resize

class WandbLogger(object):
    def __init__(
            self,
            project: str = 'vsr',
            save_dir: str = '.',
            id: str = 'sanity',
            name: str = 'Sanity Checking',
            tags: List[Any] = None
                 ):

        self.save_dir = save_dir
        self.project = project
        self.name = name
        self.id = id
        self.tags = tags

    def init(self):
        wandb.init(
            dir = self.save_dir,
            project = self.project,
            name = self.name,
            id = self.id,
            tags = self.tags,
        )

    @staticmethod
    def log_images(stage, epoch, lr, sr, hr, lq=None):
        n, t, d, h, w = hr.size()

        lr = resize(lr[0, -1, :, :, :], (h,w)).detach()
        hr = hr[0, -1, :, :, :].detach()
        sr = sr[0, -1, :, :, :].detach().clamp(0, 1)

        if lq is not None:
            lq = resize(lq[0, -1, :, :, :], (h,w)).detach()
            grid = make_grid([lr, lq, sr, hr], nrow=4, ncol=1)

        else:
            grid = make_grid([lr, sr, hr], nrow=3, ncol=1)

        wandb.log({f'Prediction {stage}': [wandb.Image(grid, caption=f'Stage {stage}, Epoch {epoch}')]})

    @staticmethod
    def log_dict(log_dict, stage="Train"):
        out_dict = {}
        for key in log_dict.keys():
            out_dict['/'.join([key, stage])] = log_dict[key]

        wandb.log(out_dict)

    @staticmethod
    def close():
        wandb.finish()
