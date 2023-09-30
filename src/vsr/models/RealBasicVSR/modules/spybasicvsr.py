import logging

import torch
import torch.nn as nn
from vsrlab.core.modules.conv import ResidualBlock
from vsrlab.core.modules.upsampling import PixelShufflePack
from vsrlab.optical_flow.models.spynet.model import SpyNet
from vsrlab.vsr.models.RealBasicVSR.modules.spynet import flow_warp

pylogger = logging.getLogger(__name__)

class SPyBasicVSR(nn.Module):
    def __init__(self, mid_channels=64, res_blocks=30, upscale=4, k=5,
                 pretrained_flow=None, train_flow=False):
        super().__init__()
        self.mid_channels = mid_channels
        self.backward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.forward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.point_conv = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 1, 1), nn.LeakyReLU(0.1))
        self.upsample = nn.Sequential(*[PixelShufflePack(mid_channels, mid_channels, 2) for _ in range(upscale // 2)])
        self.conv_last = nn.Sequential(nn.Conv2d(mid_channels, 64, 3, 1, 1), nn.LeakyReLU(0.1),
                                       nn.Conv2d(64, 3, 3, 1, 1))
        self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)
        self.spynet = SpyNet.from_pretrained(k, [4], pretrained_flow)

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if not train_flow:
            print('setting optical flow weights to no_grad')
            for param in self.spynet.parameters():
                param.requires_grad = False

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()

        lrs_1 = (lrs[:, :-1, :, :, :].reshape(-1, c, h, w) - self.mean) / self.std  # remove last frame
        lrs_2 = (lrs[:, 1:, :, :, :].reshape(-1, c, h, w) - self.mean) / self.std  # remove first frame

        flow_backward = self.spynet((lrs_1, lrs_2), train=False)
        flow_forward = self.spynet((lrs_2, lrs_1), train=False)

        return flow_forward, flow_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        flow_forward, flow_backward = self.compute_flow(lrs)
        flows_forward = flow_forward.view(n, t - 1, 2, h, w)
        flows_backward = flow_backward.view(n, t - 1, 2, h, w)

        outputs = []  # backward-propagation
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            # no warping required for the last timestep
            if i < t - 1:
                # (b c h w)
                flow = flows_backward[:, i, :, :, :]
                # propagated frame
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # mid_ch + 3
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            # (b mid_ch h w)
            feat_prop = self.backward_resblocks(feat_prop)
            # t -> (b mid_ch h w)
            outputs.append(feat_prop)

        outputs = outputs[::-1]  # forward-propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            # no warping required for the first timestep
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            # mid_ch + 3
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            # (b mid_ch h w)
            feat_prop = self.forward_resblocks(feat_prop)
            # (b mid_ch*2 h w)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            # (b mid_ch h w)
            out = self.point_conv(out)
            # (b mid_ch h*u w*u)
            out = self.upsample(out)
            # (b 3 h w)
            out = self.conv_last(out)
            outputs[i] = out + self.upscale(lrs[:, i, :, :, :])
        return torch.stack(outputs, dim=1)

def main() -> None:
    model = BasicVSR(4, 1)
    spy_params = list(filter(lambda kv: "spynet" in kv[0], model.named_parameters()))
    base_params = list(filter(lambda kv: not "spynet" in kv[0], model.named_parameters()))
    print([i[0] for i in spy_params])
    print([i[0] for i in base_params])

if __name__ == "__main__":
    main()