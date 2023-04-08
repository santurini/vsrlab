from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from core import PROJECT_ROOT
from core.modules.conv import ConvReLU
from torch.nn.functional import grid_sample

class SpynetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(ConvReLU(8, 32, 7, 1, 3), ConvReLU(32, 64, 7, 1, 3),
                                          ConvReLU(64, 32, 7, 1, 3), ConvReLU(32, 16, 7, 1, 3),
                                          ConvReLU(16, 2, 7, 1, 3))

    def forward(self, x):
        return self.basic_module(x)

class Spynet(nn.Module):
    def __init__(
            self,
            pretrained=f'{PROJECT_ROOT}/src/optical_flow/weights/pretrained_spynet.pth'
    ):
        super().__init__()
        self.basic_module = nn.ModuleList([SpynetModule() for _ in range(6)])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained)
            new_dict = OrderedDict([(key[13:34] + '.0' + key[34:], state_dict[key]) for key in state_dict.keys()])
            self.basic_module.load_state_dict(new_dict)
            print('LOADED SPYNET PRETRAINED WEIGHTS')

    def compute_flow(self, ref, supp):
        t, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for level in range(5):
            # downsample frames up to a factor of 32
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))

        flows = []
        ref = ref[::-1];
        supp = supp[::-1]  # start from smallest scale
        flow = ref[0].new_zeros(t, 2, h // 32, w // 32)  # tensor of all zeros with (H, W) of stride 32
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0  # why?

            flow_residue = self.basic_module[level](  # input_channel = 8 -> out_channel = 2
                torch.cat([ref[level],  # 3 channels
                           flow_warp(supp[level],
                                     flow_up.permute(0, 2, 3, 1),
                                     padding_mode='border'),  # 3 channels
                           flow_up], 1  # 2 channels
                          )
            )

            flow = flow_up + flow_residue
            flows.append(flow)

        return flows

    def forward(self, ref, supp):
        return self.compute_flow(ref, supp)

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    t, c, h, w = x.size()  # -> flow has channels last shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False
    grid_flow = grid + flow  # broadcast on frames dimension, 1st round everyone in same location
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0  # min-max normalize in [-1, 1]
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)  # (t, h, w, 2)
    # given input and flow grid computes output using input values and pixel locations from grid
    output = grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output  # 3-channel frames of same spatial resolution as x
