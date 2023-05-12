from typing import Any, List

import torch
import wandb
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import OmegaConf
from optical_flow.flow_viz import flow_tensor_to_image
from torchvision.utils import make_grid

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

    def init(self, cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        self.run = wandb.init(
            dir=self.save_dir,
            project=self.project,
            name=self.name,
            id=self.id,
            tags=self.tags,
            config=cfg
        )

    def log_images(self, stage, epoch, lr, sr, hr, lq=None):
        n, t, d, h, w = hr.size()

        lr = resize(lr[0, -1, :, :, :], (h, w)).detach().cpu()
        hr = hr[0, -1, :, :, :].detach().cpu()
        sr = sr[0, -1, :, :, :].detach().clamp(0, 1).cpu()

        if lq is not None:
            lq = resize(lq[0, -1, :, :, :], (h, w)).detach().cpu()
            grid = make_grid([lr, lq, sr, hr], nrow=4, ncol=1)

        else:
            grid = make_grid([lr, sr, hr], nrow=3, ncol=1)

        self.run.log({f'Prediction {stage}': [wandb.Image(grid, caption=f'Stage {stage}, Epoch {epoch}')]})

    def log_flow(self, stage, epoch, inputs, cleaned, flow, gt_flow):
        x1 = inputs[0].detach().cpu()
        x2 = inputs[1].detach().cpu()
        cleaned = cleaned[0].detach().cpu()
        flow_viz = torch.from_numpy(flow_tensor_to_image(flow[0].detach().cpu()))
        gt_flow = torch.from_numpy(flow_tensor_to_image(gt_flow[0].detach().cpu()))
        grid = make_grid([cleaned, x1, x2, flow_viz, gt_flow], nrow=5, ncol=1)
        self.run.log({f'Flow {stage}': [wandb.Image(grid, caption=f'Epoch {epoch}')]})

    def log_dict(self, log_dict, epoch, stage="Train"):
        out_dict = {}
        for key in log_dict.keys():
            out_dict['/'.join([key, stage])] = log_dict[key]

        self.run.log(out_dict | {"epoch": epoch})

    def save(self, save_path, base_path):
        self.run.save(glob_str=save_path, base_path=base_path)

    @staticmethod
    def close():
        wandb.finish()
