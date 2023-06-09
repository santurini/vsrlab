import logging
import os

import torch
import torch.nn as nn
from core.modules.conv import ResidualBlock
from core.modules.upsampling import PixelShufflePack
from fmoe.gates.gshard_gate import GShardGate
from vsr.models.MoEVRT.tmsa_v2 import LinearMoE
from vsr.models.RealBasicVSR.modules.spynet import Spynet, flow_warp
from vsr.models.VRT.modules.deform_conv import DCNv2PackFlowGuided

pylogger = logging.getLogger(__name__)

class BasicVSR(nn.Module):
    def __init__(
            self,
            num_experts=4,
            top_k=2,
            num_gpus=1,
            gate=GShardGate,
            deformable_groups=4,
            mid_channels=64,
            res_blocks=30,
            upscale=4,
            pretrained_flow=False,
            train_flow=False
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.conv_first = nn.Conv3d(3, mid_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.backward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.forward_resblocks = ResidualBlock(mid_channels + 3, mid_channels, res_blocks)
        self.point_conv = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 1, 1), nn.LeakyReLU(0.1))
        self.upsample = nn.Sequential(*[PixelShufflePack(mid_channels, mid_channels, 2) for _ in range(upscale // 2)])
        self.conv_last = nn.Sequential(nn.Conv2d(mid_channels, 64, 3, 1, 1), nn.LeakyReLU(0.1),
                                       nn.Conv2d(64, 3, 3, 1, 1))
        self.upscale = nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False)
        self.spynet = Spynet(pretrained_flow)

        # parallel warping
        self.pa_deform = DCNv2PackFlowGuided(mid_channels, mid_channels, 3, padding=1,
                                             deformable_groups=deformable_groups,
                                             max_residue_magnitude=10, pa_frames=2)
        self.pa_fuse = LinearMoE(
            num_expert=num_experts,
            in_features=mid_channels,
            hidden_features=mid_channels,
            act_layer=nn.GELU,
            expert_rank=os.environ.get("OMPI_COMM_WORLD_RANK", 0),
            world_size=num_gpus,
            top_k=top_k,
            gate=gate,
            expert_dp_comm="dp" if num_gpus > 1 else "none"
        )

        if not train_flow:
            pylogger.info('Setting Optical Flow weights to no_grad')
            for param in self.spynet.parameters():
                param.requires_grad = False

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flow_backward = self.spynet(lrs_1, lrs_2)
        if self.is_mirror:
            flow_forward = None
        else:
            flow_forward = self.spynet(lrs_2, lrs_1)

        return flow_forward, flow_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        features = self.conv_first(lrs)
        flow_forward, flow_backward = self.compute_flow(lrs)
        flows_forward = flow_forward.view(n, t - 1, 2, h, w)
        flows_backward = flow_backward.view(n, t - 1, 2, h, w)

        outputs = []  # backward-propagation
        for i in range(t - 1, 0, -1):
            x_i = features[:, i, ...]
            x_prev = features[:, i - 1, ...]
            flow = flows_backward[:, i - 1, :, :, :]
            feat_prop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
            feat_prop = self.pa_deform(x_i, [feat_prop], x_prev, [flow])
            feat_prop = self.pa_fuse(torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1))
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)

        outputs = outputs[::-1]  # forward-propagation
        feat_prop = features[:, 0, ...]
        for i in range(0, t):
            if i > 0:
                x_i = features[:, i - 1, ...]
                x_next = features[:, i, ...]
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = self.pa_deform(x_i, [feat_prop], x_next, [flow])
                feat_prop = self.pa_fuse(torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.point_conv(out)
            out = self.upsample(out)
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
