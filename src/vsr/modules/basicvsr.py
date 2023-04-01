import torch
import torch.nn as nn
from core.modules.conv import ResidualBlock
from core.modules.upsampling import PixelShufflePack
from optical_flow.modules.spynet import Spynet, flow_warp

class BasicVSR(nn.Module):
    def __init__(self, mid_channels=64, res_blocks=30, upscale=4, is_mirror=False):
        super().__init__()
        self.name = 'BasicVSR'
        self.is_mirror = is_mirror
        self.mid_channels = mid_channels
        self.spynet = Spynet().requires_grad_(False)
        self.backward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.forward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.point_conv = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 1, 1), nn.LeakyReLU(0.1))
        self.upsample = nn.Sequential(*[PixelShufflePack(mid_channels, mid_channels, 2) for _ in range(upscale // 2)])
        self.conv_last = nn.Sequential(nn.Conv2d(mid_channels, 64, 3, 1, 1), nn.LeakyReLU(0.1),
                                       nn.Conv2d(64, 3, 3, 1, 1))
        self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        flows_forward, flows_backward = self.compute_flow(lrs)

        outputs = []  # backward-propagation
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]  # b c h w
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))  # propagated frame

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)  # mid_ch + 3
            feat_prop = self.backward_resblocks(feat_prop)  # (b mid_ch h w)
            outputs.append(feat_prop)  # t -> b mid_ch h w

        outputs = outputs[::-1]  # forward-propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            if i > 0:  # no warping required for the first timestep
                if self.is_mirror:
                    flow = flows_backward[:, -i, :, :, :]
                else:
                    flow = flows_forward[:, i - 1, :, :, :]  # flow at previous frame (?)
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)  # mid_ch + 3
            feat_prop = self.forward_resblocks(feat_prop)  # b mid_ch h w
            out = torch.cat([outputs[i], feat_prop], dim=1)  # b mid_ch*2 h w
            out = self.point_conv(out)  # b mid_ch h w
            out = self.upsample(out)  # b mid_ch h*u w*u
            out = self.conv_last(out)  # b 3 h w
            outputs[i] = out + self.upscale(lrs[:, i, :, :, :])
        return torch.stack(outputs, dim=1)