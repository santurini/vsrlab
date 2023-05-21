from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from core import PROJECT_ROOT
from optical_flow.models import spynet

class BasicModule(nn.Module):
    def __init__(self, input_channels: int = 8):
        super(BasicModule, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=7, padding=3, stride=1))

    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor], 
                optical_flow: torch.Tensor = None,
                upsample_optical_flow: bool = True) -> torch.Tensor:
        f_frame, s_frame = frames

        if optical_flow is None:
            # If optical flow is None (k = 0) then create empty one having the
            # same size as the input frames, therefore there is no need to 
            # upsample it later
            upsample_optical_flow = False
            b, c, h, w = f_frame.size()
            optical_flow = torch.zeros(b, 2, h, w, device=s_frame.device)

        if upsample_optical_flow:
            optical_flow = F.interpolate(
                optical_flow, scale_factor=2, align_corners=True, 
                mode='bilinear') * 2

        s_frame = spynet.nn.warp(s_frame, optical_flow, s_frame.device)
        s_frame = torch.cat([s_frame, optical_flow], dim=1)
        
        inp = torch.cat([f_frame, s_frame], dim=1)
        return self.module(inp)


class SpyNet(nn.Module):

    def __init__(
            self,
            units: Sequence[BasicModule] = None,
            k: int = None,
            return_levels=[2, 3, 4]
    ):
        super(SpyNet, self).__init__()

        if units is not None and k is not None:
            assert len(units) == k

        if units is None and k is None:
            raise ValueError('At least one argument (units or k) must be'
                             'specified')

        if units is not None:
            self.units = nn.ModuleList(units)
        else:
            units = [BasicModule() for _ in range(k)]
            self.units = nn.ModuleList(units)

        self.levels = k - 1
        self.return_levels = return_levels

    def forward(self,
                frames: Tuple[torch.Tensor, torch.Tensor],
                train: bool = True,
                limit_k: int = -1) -> torch.Tensor:
        """
        Parameters
        ----------
        frames: Tuple[torch.Tensor, torch.Tensor]
            Highest resolution frames. Each tuple element has shape
            [BATCH, 3, HEIGHT, WIDTH]
        """
        flow_list = []

        if limit_k == -1:
            units = self.units
        else:
            units = self.units[:limit_k]
        Vk_1 = None

        for k, G in enumerate(units):
            b, c, h, w = frames[0].shape
            if not train:
                im_size = (
                    (h // 2 ** (self.levels - k)),
                    (w // 2 ** (self.levels - k))
                )
            else:
                im_size = spynet.config.GConf(k).image_size

            x1 = F.interpolate(frames[0], im_size, mode='bilinear',
                               align_corners=True)
            x2 = F.interpolate(frames[1], im_size, mode='bilinear',
                               align_corners=True)

            if Vk_1 is not None:  # Upsample the previous optical flow
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, align_corners=True,
                    mode='bilinear') * 2.

            Vk = G((x1, x2), Vk_1, upsample_optical_flow=False)
            Vk_1 = Vk + Vk_1 if Vk_1 is not None else Vk

            if k in self.return_levels:
                print(f"level {k} -> shape {Vk_1.shape}")
                flow_list.insert(0, Vk_1)

        if len(flow_list) == 1:
            return flow_list[0]

        return flow_list

    @classmethod
    def from_pretrained(cls: Type['SpyNet'], k, path=None) -> 'SpyNet':

        def get_model(ckpt_path: str, levels) -> 'SpyNet':
            # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['params']
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            checkpoint = {
                k.replace('basic_module', 'units', 1).replace('basic_module', 'module'): v
                for k, v in checkpoint.items()
            }

            instance = cls(k=levels)
            instance.load_state_dict(checkpoint, strict=False)

            return instance

        if path is None:
            path = f'{PROJECT_ROOT}/src/vsr/models/VRT/weights/spynet_sintel_final-3d2a1287.pth'

        return get_model(path, k)
