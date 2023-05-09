import logging

import torch
import torch.nn as nn
from core.modules.conv import ResidualBlock
from core.modules.upsampling import PixelShufflePack
from optical_flow.models.irr.irr import IRRPWCNet
from optical_flow.models.spynet import Spynet, flow_warp

pylogger = logging.getLogger(__name__)

class BasicVSR(nn.Module):
    def __init__(self, mid_channels=64, res_blocks=30, upscale=4, is_mirror=False, optical_flow='spynet',
                 pretrained_flow=False, train_flow=False):
        super().__init__()
        self.is_mirror = is_mirror
        self.mid_channels = mid_channels
        self.backward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.forward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.point_conv = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 1, 1), nn.LeakyReLU(0.1))
        self.upsample = nn.Sequential(*[PixelShufflePack(mid_channels, mid_channels, 2) for _ in range(upscale // 2)])
        self.conv_last = nn.Sequential(nn.Conv2d(mid_channels, 64, 3, 1, 1), nn.LeakyReLU(0.1),
                                       nn.Conv2d(64, 3, 3, 1, 1))
        self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)

        self.optical_flow_name = optical_flow
        if optical_flow == "spynet":
            self.spynet = Spynet(pretrained_flow)
        elif optical_flow == "irr":
            self.irr = IRRPWCNet(pretrained_flow, [-1])
        else:
            raise Exception("Not a valid optical flow, possible options are: spynet, irr")

        if not train_flow:
            pylogger.info('Setting Optical Flow weights to no_grad')
            for param in getattr(self, f"{optical_flow}").parameters():
                param.requires_grad = False

    def compute_flow_spynet(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flow_backward = self.spynet(lrs_1, lrs_2)
        if self.is_mirror:
            flow_forward = None
        else:
            flow_forward = self.spynet(lrs_2, lrs_1)

        return flow_forward, flow_backward

    def compute_flow_irr(self, lrs):
        '''Get flow between frames t and t+1 from x.'''
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame

        flow_forward, flow_backward = self.irr(lrs_2, lrs_1)

        return flow_forward[0], flow_backward[0]

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        flow_forward, flow_backward = getattr(self, f'compute_flow_{self.optical_flow_name}')(lrs)
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
                if self.is_mirror:
                    flow = flows_backward[:, -i, :, :, :]
                else:
                    # flow at previous frame (?)
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
