import logging
import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from optical_flow.models.irr.irr import IRRPWCNet
from core.losses import CharbonnierLoss

from vsr.models.VRT.modules.spynet import SpyNet, flow_warp
from vsr.models.VRT.modules.stage import Stage
from vsr.models.VRT.modules.tmsa import RTMSA

pylogger = logging.getLogger(__name__)

loss_fn = CharbonnierLoss()

class Transpose_Dim12(nn.Module):
    """ Transpose Dim1 and Dim2 of a tensor."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.transpose(1, 2)

class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert (scale & (scale - 1)) == 0, f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.'

        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(2))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        super().__init__(*m)

class VRT(nn.Module):
    """ Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
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
        optical_flow_path (str): Pretrained optical_flow model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 out_chans=3,
                 img_size=[6, 64, 64],
                 window_size=[6, 8, 8],
                 depths=[4, 4, 4, 4, 4, 4, 4, 2, 2],
                 indep_reconsts=[11, 12],
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
                 recal_all_flows=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows

        # conv_first
        conv_first_in_chans = in_chans*(1+2*4)
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.optical_flow_name = optical_flow
        if optical_flow == "spynet":
            self.optical_flow = SpyNet(optical_flow_pretrained, [2, 3, 4, 5])
        elif optical_flow == "irr":
            self.optical_flow = IRRPWCNet(optical_flow_pretrained, [-1, -2, -3, -4])
        else:
            raise Exception("Not a valid optical flow, possible options are: spynet, irr")

        pylogger.info(f'Initialized Optical Flow module as: <{self.optical_flow.__class__.__name__}>')

        if not optical_flow_train:
            pylogger.info(f'Freezing Optical Flow parameters')
            for p in self.optical_flow.parameters():
                p.requires_grad = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]

        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(len(depths))]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(len(depths))]

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
                        max_residue_magnitude=10 / scales[i],
                        use_checkpoint_attn=use_checkpoint_attns[i],
                        use_checkpoint_ffn=use_checkpoint_ffns[i],
                        )
                    )

        # stage 8
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
                Rearrange('n c d h w ->  n d h w c'),
                nn.LayerNorm(embed_dims[6]),
                nn.Linear(embed_dims[6], embed_dims[7]),
                Rearrange('n d h w c -> n c d h w')
            )]
        )

        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer,
                      use_checkpoint_attn=use_checkpoint_attns[i],
                      use_checkpoint_ffn=use_checkpoint_ffns[i]
                      )
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

    def forward(self, lr):
        # x: (N, D, C, H, W)
        x_lq = lr.clone()

        # calculate flows
        flows_backward, flows_forward = getattr(self, f'get_flows_{self.optical_flow_name}')(lr)

        # warp input
        x_backward, x_forward = self.get_aligned_image(lr,  flows_backward[0], flows_forward[0])
        x = torch.cat([lr, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)

        _, _, C, H, W = x.shape
        upscale = torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)
        sr = x + upscale

        return sr, lr

    def train_step(self, lr, hr):
        # x: (N, D, C, H, W)
        x_lq = lr.clone()

        # calculate flows
        flows_backward, flows_forward = getattr(self, f'get_flows_{self.optical_flow_name}')(lr)

        # warp input
        x_backward, x_forward = self.get_aligned_image(lr,  flows_backward[0], flows_forward[0])
        x = torch.cat([lr, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)

        _, _, C, H, W = x.shape
        upscale = torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)

        sr = x + upscale

        loss = loss_fn(sr, hr)

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

        # backward
        flows_backward = self.optical_flow(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(4))]

        # forward
        flows_forward = self.optical_flow(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(4))]

        return flows_backward, flows_forward

    def get_flows_irr(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward, flows_backward = self.optical_flow(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(4))]
        flows_backward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(4))]

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

@torch.no_grad()
def main() -> None:
    model = VRT()
    x = torch.rand(2, 6, 3, 64, 64)
    print(model(x).shape)

if __name__ == "__main__":
    main()