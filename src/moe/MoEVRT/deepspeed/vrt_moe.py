import logging
import math
import os
from distutils.version import LooseVersion

import deepspeed
import torch
import torch.nn as nn
from core.losses import CharbonnierLoss
from core.modules.conv import ResidualBlock
from einops import rearrange
from einops.layers.torch import Rearrange
from vsr.models.VRT.modules.spynet import SpyNet, flow_warp
from vsr.models.VRT.modules.stage import Stage
from vsr.models.VRT.modules.tmsa import RTMSA

pylogger = logging.getLogger(__name__)

loss_fn = CharbonnierLoss()

class Debug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if int(os.environ["RANK"]) == 0:
            print("IM HERE:", x.shape)
        return x

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        assert (scale & (scale - 1)) == 0, "Scale should be a power of 2"

        class Transpose_Dim12(nn.Module):
            """ Transpose Dim1 and Dim2 of a tensor."""

            def __init__(self):
                super().__init__()

            @staticmethod
            def forward(x):
                return x.transpose(1, 2)

        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(2))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        super(Upsample, self).__init__(*m)

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks, steps):
        super().__init__()
        self.steps = steps
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        n, t, c, h, w = x.size()
        for _ in range(self.steps):  # at most 3 cleaning, determined empirically
            x = x.view(-1, c, h, w)
            residues = self.conv(self.resblock(x))
            x = (x + residues).view(n, t, c, h, w)
        return x

class TinyVRT(nn.Module):
    def __init__(
            self,
            upscale=4,
            in_chans=3,
            out_chans=3,
            refine_steps=3,
            refine_blocks=5,
            refine_ch=64,
            img_size=[6, 64, 64],
            window_size=[6, 8, 8],
            depths=[8, 8, 8, 8, 8, 4, 4],
            indep_reconsts=[-2, -1],
            embed_dims=[64, 64, 64, 64, 64, 80, 80],
            num_heads=[6, 6, 6, 6, 6, 6, 6],
            num_experts=2,
            num_gpus=2,
            top_k=1,
            mul_attn_ratio=0.75,
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            optical_flow_pretrained=True,
            optical_flow_train=False,
            pa_frames=2,
            deformable_groups=8
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.indep_reconsts = [i.item() for i in torch.arange(len(depths))[indep_reconsts]]

        self.iterative_refinement = IterativeRefinement(
            refine_ch,
            refine_blocks,
            refine_steps
        )

        # conv_first
        conv_first_in_chans = in_chans * (1 + 2 * 4)
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.init_flow(optical_flow_pretrained, optical_flow_train)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        reshapes = ['none', 'down', 'down', 'up', 'up']
        scales = [1, 2, 4, 2, 1]

        # stage 1-5
        for i in range(len(scales)):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=embed_dims[i - 1],
                        dim=embed_dims[i],
                        input_resolution=(img_size[0], img_size[1] // scales[i], img_size[2] // scales[i]),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        mul_attn_ratio=mul_attn_ratio,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i]
                    )
                    )

        # last stage
        self.stage6 = deepspeed.moe.layer.MoE(
            hidden_size=embed_dims[len(scales) - 1],
            expert=nn.Sequential(*
                                 [
                                     Debug(),
                                     nn.LayerNorm(embed_dims[len(scales) - 1]),
                                     Debug(),
                                     nn.Linear(embed_dims[len(scales) - 1], embed_dims[len(scales)]),
                                     Debug(),
                                     Rearrange('n g (d h w) c -> n (c g) d h w', d=6, w=img_size[1]),
                                     Debug(),
                                 ] +
                                 [
                                     RTMSA(dim=embed_dims[i],
                                           input_resolution=img_size,
                                           depth=depths[i],
                                           num_heads=num_heads[i],
                                           window_size=[1, window_size[1],
                                                        window_size[2]] if i in self.indep_reconsts else window_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                           norm_layer=norm_layer
                                           ) for i in range(len(scales), len(depths))
                                 ]
                                 ),
            num_experts=num_experts,
            ep_size=num_gpus,
            k=top_k
        )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True))

        self.upsample = Upsample(upscale, num_feat)

        self.conv_last = nn.Conv3d(num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        # x: (N, D, C, H, W)
        x_lq = self.iterative_refinement(x)

        # calculate flows
        flows_backward, flows_forward = self.get_flows(x)

        # warp input
        x_backward, x_forward = self.get_aligned_image(x, flows_backward[0], flows_forward[0])
        x = torch.cat([x, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(
            1, 4)
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
        _, _, C, H, W = x.shape

        sr = x + torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)

        return sr, x_lq

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::3], flows_forward[0::3])  # =
        x2 = self.stage2(x1, flows_backward[1::3], flows_forward[1::3])  # stride 2
        x3 = self.stage3(x2, flows_backward[2::3], flows_forward[2::3])  # stride 4
        x = self.stage4(x3, flows_backward[1::3], flows_forward[1::3])  # stride 2
        x = self.stage5(x + x2, flows_backward[0::3], flows_forward[0::3])  # =
        x = rearrange(x + x1, 'n c d h w ->  n d h w c')

        x, _, _ = self.stage6(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def get_flows(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        n_scales = len(self.optical_flow.return_levels)

        # backward
        flows_backward = self.optical_flow(x_1, x_2)
        flows_backward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                          zip(flows_backward, range(n_scales))]

        # forward
        flows_forward = self.optical_flow(x_2, x_1)
        flows_forward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(n_scales))]

        return flows_backward, flows_forward

    @staticmethod
    def get_aligned_image(x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''
        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))  # frame i+1 aligned towards i

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))  # frame i-1 aligned towards i

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def init_flow(self, pretrained, train):
        self.optical_flow = SpyNet(pretrained, [3, 4, 5])

        if not train:
            pylogger.info(f'Freezing Optical Flow parameters')
            for p in self.optical_flow.parameters():
                p.requires_grad = False

@torch.no_grad()
def main() -> None:
    model = TinyVRT(
        upsample=4,
        in_chans=3,
        out_chans=3,
        refine_steps=3,
        refine_blocks=5,
        refine_ch=64,
        img_size=[6, 64, 64],
        window_size=[6, 8, 8],
        depths=[8, 8, 8, 8, 8, 4, 4],
        indep_reconsts=[-2, -1],
        embed_dims=[64, 64, 64, 64, 64, 80, 80],
        num_heads=[6, 6, 6, 6, 6, 6, 6],
        num_experts=4,
        num_gpus=2,
        top_k=2,
        mul_attn_ratio=0.75,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        optical_flow_pretrained=True,
        pa_frames=2,
        deformable_groups=8
    )

    x = torch.rand(2, 6, 3, 64, 64)
    print(model(x).shape)

if __name__ == "__main__":
    main()
