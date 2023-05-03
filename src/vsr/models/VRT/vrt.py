import logging
import math

import torch
import torch.nn as nn
from distutils.version import LooseVersion
from einops import rearrange
from einops.layers.torch import Rearrange
from kornia.geometry.transform import resize

from optical_flow.models.irr.irr import IRRPWCNet
from core.losses import CharbonnierLoss
from core.modules.conv import ResidualBlock

from vsr.models.VRT.modules.spynet import SpyNet, flow_warp
from vsr.models.VRT.modules.stage import Stage
from vsr.models.VRT.modules.tmsa import RTMSA
from vsr.models.VRT.modules.siren import SirenNet


pylogger = logging.getLogger(__name__)

loss_fn = CharbonnierLoss()

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

class VRT(nn.Module):
    """ Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        optical_flow (str): Optical Flow Model. Default: "spynet".
        optical_flow_pretrained (bool): Pretrained Optical Flow. Default: True.
        optical_flow_train (bool): Freeze or not Optical Flow. Default: True.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        restore_hidden (int): Number of hidden layers of final layer. Default: 128.
        restore_layers: Number of layers in final layer. Default: 5.
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=3,
                 refine_steps=3,
                 refine_blocks=5,
                 refine_ch=64,
                 img_size=[6, 64, 64],
                 window_size=[6, 8, 8],
                 depths=[4, 4, 4, 4, 4, 4, 4, 2, 2],
                 indep_reconsts=[-2, -1],
                 embed_dims=[32, 32, 32, 32, 32, 32, 32, 64, 64],
                 num_heads=[4, 4, 4, 4, 4, 4, 4, 4, 4],
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 optical_flow="spynet",
                 optical_flow_pretrained=True,
                 optical_flow_train=False,
                 pa_frames=2,
                 deformable_groups=4,
                 restore_hidden = 128,
                 restore_layers = 5,
                 ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.pa_frames = pa_frames
        self.indep_reconsts = [i.item() for i in torch.arange(len(depths))[indep_reconsts]]
        self.rcl = Rearrange('b c t h w -> b t h w c')
        self.rwl = Rearrange('b t h w c -> b c t h w')

        self.iterative_refinement = IterativeRefinement(
            refine_ch,
            refine_steps,
            refine_blocks,
        )

        # conv_first
        conv_first_in_chans = in_chans*(1+2*4)
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.init_flow(optical_flow, optical_flow_pretrained, optical_flow_train)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]

        # stage 1- 7
        for i in range(7):
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

        # stage 8
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
                self.rcl,
                nn.LayerNorm(embed_dims[6]),
                nn.Linear(embed_dims[6], embed_dims[7]),
                self.rwl
            )]
        )

        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in self.indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.mlp_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        self.restore = SirenNet(
            dim_in = embed_dims[0],
            dim_hidden = restore_hidden,
            dim_out = 3,
            num_layers = restore_layers,
            w0 = 1.,
            w0_initial = 30.,
            use_bias = True
        )

    def forward(self, x):
        # x: (N, D, C, H, W)
        # refine image
        lq = self.iterative_refinement(x)

        # calculate flows
        flows_backward, flows_forward = getattr(self, f'get_flows_{self.optical_flow_name}')(lq)

        # warp input
        x_backward, x_forward = self.get_aligned_image(lq,  flows_backward[0], flows_forward[0])
        x = torch.cat([lq, x_backward, x_forward], 2)

        # video restoration
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.rwl(self.mlp_after_body(self.rcl(self.forward_features(x, flows_backward, flows_forward))))
        sr = self.rwl(self.restore(self.rcl(x))).transpose(1, 2) + lq

        return sr, lr

    def train_step(self, x, hr):
        # x: (N, D, C, H, W)
        # refine image
        lq = self.iterative_refinement(x)

        # calculate flows
        flows_backward, flows_forward = getattr(self, f'get_flows_{self.optical_flow_name}')(lq)

        # warp input
        x_backward, x_forward = self.get_aligned_image(lq,  flows_backward[0], flows_forward[0])
        x = torch.cat([lq, x_backward, x_forward], 2)

        # video restoration
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.rwl(self.mlp_after_body(self.rcl(self.forward_features(x, flows_backward, flows_forward))))
        sr = self.rwl(self.restore(self.rcl(x))).transpose(1, 2) + lq

        loss = loss_fn(sr, hr) + loss_fn(lq, resize(hr, (h, w), antialias=True))

        return {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "loss": loss
        }

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4]) # =
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4]) # stride 2
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4]) # stride 4
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4]) # stride 8
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4]) # stride 4
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4]) # stride 2
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4]) # =
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def get_flows_spynet(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        n_scales = len(self.optical_flow.return_levels)

        # backward
        flows_backward = self.optical_flow(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(n_scales))]

        # forward
        flows_forward = self.optical_flow(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(n_scales))]

        return flows_backward, flows_forward

    def get_flows_irr(self, x):
        '''Get flow between frames t and t+1 from x.'''
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        n_scales = len(self.optical_flow.return_levels)
        flows_forward, flows_backward = self.optical_flow(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(n_scales))]
        flows_backward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(n_scales))]

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
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i+1 aligned towards i

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i-1 aligned towards i

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def init_flow(self, optical_flow, pretrained, train):
        self.optical_flow_name = optical_flow
        if optical_flow == "spynet":
            self.optical_flow = SpyNet(pretrained, [3, 4, 5])
        elif optical_flow == "irr":
            self.optical_flow = IRRPWCNet(pretrained, [-2, -3, -4])
        else:
            raise Exception("Not a valid optical flow, possible options are: spynet, irr")

        pylogger.info(f'Initialized Optical Flow module as: <{self.optical_flow.__class__.__name__}>')

        if not train:
            pylogger.info(f'Freezing Optical Flow parameters')
            for p in self.optical_flow.parameters():
                p.requires_grad = False

class TinyVRT(VRT):
    def __init__(
            self,
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
            mul_attn_ratio=0.75,
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            optical_flow="spynet",
            optical_flow_pretrained=True,
            optical_flow_train=False,
            pa_frames=2,
            deformable_groups=8,
            restore_hidden=128,
            restore_layers=5,
                ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.pa_frames = pa_frames
        self.indep_reconsts = [i.item() for i in torch.arange(len(depths))[indep_reconsts]]
        self.rcl = Rearrange('b c t h w -> b t h w c')
        self.rwl = Rearrange('b t h w c -> b c t h w')

        self.iterative_refinement = IterativeRefinement(
            refine_ch,
            refine_steps,
            refine_blocks,
        )

        # conv_first
        conv_first_in_chans = in_chans*(1+2*4)
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.init_flow(optical_flow, optical_flow_pretrained, optical_flow_train)

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
        self.stage6 = nn.ModuleList([
                    nn.Sequential(Rearrange('n c d h w ->  n d h w c'),
                                  nn.LayerNorm(embed_dims[len(scales)-1]),
                                  nn.Linear(embed_dims[len(scales)-1], embed_dims[len(scales)]),
                                  Rearrange('n d h w c -> n c d h w'))
                              ]
                             )

        for i in range(len(scales), len(depths)):
            self.stage6.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in self.indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.mlp_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        self.restore = SirenNet(
            dim_in = embed_dims[0],
            dim_hidden = restore_hidden,
            dim_out = 3,
            num_layers = restore_layers,
            w0 = 1.,
            w0_initial = 30.,
            use_bias = True
        )

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0], flows_forward[0]) # =
        x2 = self.stage2(x1, flows_backward[1], flows_forward[1]) # stride 2
        x3 = self.stage3(x2, flows_backward[2], flows_forward[2]) # stride 4
        x = self.stage4(x3, flows_backward[1], flows_forward[1])  # stride 2
        x = self.stage5(x + x2, flows_backward[0], flows_forward[0])  # =
        x = x + x1

        for layer in self.stage6:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

@torch.no_grad()
def main() -> None:
    model = TinyVRT(
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
        mul_attn_ratio=0.75,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        optical_flow_pretrained=True,
        pa_frames=2,
        deformable_groups=8,
        restore_hidden=128,
        restore_layers=5,
    )

    x = torch.rand(2, 6, 3, 64, 64)
    print(model(x).shape)

if __name__ == "__main__":
    main()