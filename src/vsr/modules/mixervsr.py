import torch
import torch.nn as nn
import torch.nn.functional as F
from core.modules.conv import DeformBlock, ConvSTBlock
from core.modules.dct_transforms import EncoderDCT, DecoderIDCT
from core.modules.mlp import MlpMixer
from core.modules.upsampling import PixelShufflePack3D
from einops import rearrange

class MixerVSR(nn.Module):
    def __init__(self, img_size, steps, mid_ch, it_blocks, patch_size,
                 time_dim, exp, mix_blocks, conv_blocks, upscale):
        super().__init__()
        self.upscale = upscale
        height, width = img_size
        channels_dim = 3 * patch_size ** 2
        patches_dim = (height // patch_size) * (width // patch_size)
        self.cleaner = IterativeRefinement(steps, 3, mid_ch, it_blocks)  # b t c h w
        self.encoder = EncoderDCT(patch_size)  # b t (h/p)*(w/p) (c*p*p)
        self.mixer = MlpMixer(patches_dim, channels_dim, time_dim, exp, mix_blocks)  # b t (h/p)*(w/p) (c*p*p)
        self.decoder = DecoderIDCT(mid_ch, patch_size, height, width)  # b t c h w
        self.upsample = SuperResolver(mid_ch, mid_ch, conv_blocks, upscale)  # b t c u*h u*w

    def forward(self, x):
        b, t, c, h, w = x.shape
        lq = self.cleaner(x)
        x_c = rearrange(lq, 'b t c h w -> (b t) c h w')
        print(x.shape)
        x = self.encoder(lq)
        print(x.shape)
        x = self.mixer(x)
        x = self.decoder(x)
        print(x.shape)
        x = self.upsample(x)
        up = F.interpolate(x_c, scale_factor=self.upscale, mode='bilinear')
        sr = x + rearrange(up, '(b t) c h w -> b t c h w', b=b, t=t)
        return sr, lq, None, None

class IterativeRefinement(nn.Module):
    def __init__(self, steps, *args, **kwargs):
        super().__init__()
        self.dcblock = DeformBlock(*args, **kwargs)
        self.steps = steps

    def forward(self, x):
        b, t, c, h, w = x.shape
        for _ in range(self.steps):
            x = x.view(-1, c, h, w)
            res = self.dcblock(x)
            x = (x + res).view(b, t, c, h, w)
            if torch.mean(torch.abs(res)) < 1e-3:
                break
        return x

class SuperResolver(nn.Module):
    def __init__(self, in_ch, mid_ch, blocks, upscale):
        super().__init__()
        steps = upscale // 2
        self.upscale = upscale
        self.conv_block = ConvSTBlock(blocks, in_ch, mid_ch)
        self.pre_upsample = nn.Sequential(*[PixelShufflePack3D(mid_ch, mid_ch, 2) for i in range(steps - 1)])
        self.upsample = PixelShufflePack3D(mid_ch, 3, 2)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pre_upsample(x)
        x = self.upsample(x)
        return x
