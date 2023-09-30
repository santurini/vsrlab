import logging
import os

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from fmoe.gates.gshard_gate import GShardGate
from fmoe.layers import FMoE
from fmoe.linear import FMoELinear
from vsrlab.core.modules.conv import ResidualBlock
from vsrlab.core.modules.upsampling import PixelShufflePack
from vsrlab.vsr.models.RealBasicVSR.modules.spynet import Spynet, flow_warp
from vsrlab.vsr.models.VRT.modules.deform_conv import DCNv2PackFlowGuided

pylogger = logging.getLogger(__name__)

class _Expert(nn.Module):
    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.itoh = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.htoi = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        x = self.itoh(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.htoi(x, fwd_expert_count)
        return x

class MoEMLP(FMoE):
    def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs
    ):
        def one_expert(d_model):
            return _Expert(1, d_model, d_hidden, activation, rank=expert_rank)

        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)

class MLP(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.itoh = nn.Linear(d_model, d_hidden, bias=True)
        self.htoi = nn.Linear(d_hidden, d_model, bias=True)
        self.activation = nn.GELU()

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.itoh(x)
        x = self.activation(x)
        x = self.htoi(x)
        return rearrange(x, 'b h w c -> b c h w')

class BasicVSR(nn.Module):
    def __init__(
            self,
            moefy=True,
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
        self.conv_first = nn.Sequential(
            Rearrange('b t c h w -> b c t h w'),
            nn.Conv3d(3, mid_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            Rearrange('b c t h w -> b t c h w')
        )
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
        if moefy:
            self.pa_fuse = MoEMLP(
                num_expert=num_experts,
                d_model=mid_channels + 3,
                d_hidden=mid_channels * 2,
                expert_rank=os.environ.get("OMPI_COMM_WORLD_RANK", 0),
                world_size=num_gpus,
                top_k=top_k,
                gate=gate,
                expert_dp_comm="dp" if num_gpus > 1 else "none"
            )
        else:
            self.pa_fuse = MLP(
                d_model=mid_channels + 3,
                d_hidden=mid_channels * 2
            )

        if not train_flow:
            print('setting optical flow weights to no_grad')
            for param in self.spynet.parameters():
                param.requires_grad = False

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flow_backward = self.spynet(lrs_1, lrs_2)
        flow_forward = self.spynet(lrs_2, lrs_1)

        return flow_forward, flow_backward

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        features = self.conv_first(lrs)
        flow_forward, flow_backward = self.compute_flow(lrs)
        flows_forward = flow_forward.view(n, t - 1, 2, h, w)
        flows_backward = flow_backward.view(n, t - 1, 2, h, w)

        outputs = []  # backward-propagation
        feat_prop = features[:, t - 1, ...]
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                x_i = features[:, i, ...]
                x_prev = features[:, i + 1, ...]
                flow = flows_backward[:, i, ...]
                feat_prop = flow_warp(x_i, flow.permute(0, 2, 3, 1))
                feat_prop = self.pa_deform(x_i, [feat_prop], x_prev, [flow])
                feat_prop = self.pa_fuse(torch.cat([lrs[:, i, ...], feat_prop], dim=1))
            else:
                feat_prop = torch.cat([lrs[:, i, ...], feat_prop], dim=1)

            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)

        outputs = outputs[::-1]  # forward-propagation
        feat_prop = features[:, 0, ...]  # 64
        for i in range(0, t):
            if i > 0:
                x_i = features[:, i - 1, ...]
                x_next = features[:, i, ...]
                flow = flows_forward[:, i - 1, ...]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = self.pa_deform(x_i, [feat_prop], x_next, [flow])
                feat_prop = self.pa_fuse(torch.cat([lrs[:, i, ...], feat_prop], dim=1))
            else:
                feat_prop = torch.cat([lrs[:, i, ...], feat_prop], dim=1)

            feat_prop = self.forward_resblocks(feat_prop)
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.point_conv(out)
            out = self.upsample(out)
            out = self.conv_last(out)
            outputs[i] = out + self.upscale(lrs[:, i, ...])

        return torch.stack(outputs, dim=1)

def main() -> None:
    model = BasicVSR(4, 1)
    spy_params = list(filter(lambda kv: "spynet" in kv[0], model.named_parameters()))
    base_params = list(filter(lambda kv: not "spynet" in kv[0], model.named_parameters()))
    print([i[0] for i in spy_params])
    print([i[0] for i in base_params])

if __name__ == "__main__":
    main()
