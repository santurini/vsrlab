from typing import Any, List
import wandb

from torchvision.utils import make_grid
from kornia.geometry.transform import resize

class WandbLogger:
    def __init__(
            self,
            project: str = 'vsr',
            save_dir: str = '.',
            run_id: str = 'sanity',
            name: str = 'Sanity Checking',
            tags: List[Any] = None
                 ):

        self.save_dir = save_dir
        self.project = project
        self.name = name
        self.run_id = run_id
        self.tags = tags

    def init(self):
        wandb.init(
            dir = self.save_dir,
            project = self.project,
            name = self.name,
            id = self.run_id,
            tags = self.tags,
        )

    @staticmethod
    def log_images(stage, epoch, lr, sr, hr, lq=None):
        _, _, _, h, w = hr.size()

        lr = resize(lr[0, -1, :, :, :], (h,w)).detach()
        hr = hr[0, -1, :, :, :].detach()
        sr = sr[0, -1, :, :, :].detach().clamp(0, 1)

        if lq is not None:
            lq = resize(lq[0, -1, :, :, :], (h, w)).detach()
            grid = make_grid([lr, lq, sr, hr], nrow=4, ncol=1)

        else:
            grid = make_grid([lr, sr, hr], nrow=3, ncol=1)

        wandb.log({f'Prediction {stage}': [wandb.Image(grid, caption=f'Stage {stage}, Epoch {epoch}')]})

    @staticmethod
    def log_dict(log_dict, average_by=1, stage="Train"):
        out_dict = {}
        for key in log_dict.keys():
            out_dict['/'.join([key, stage.capitalize()])] = log_dict[key] / average_by

        wandb.log(out_dict)

    @staticmethod
    def close():
        wandb.finish()
