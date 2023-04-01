import math

import torch.nn as nn
from einops import rearrange
from torchvision.ops import DeformConv2d

class DeformConv(DeformConv2d):
    def __init__(self, in_ch, out_ch, kernel_size=3, deformable_groups=1):
        super().__init__(in_ch, out_ch, kernel_size)
        if in_ch % deformable_groups != 0: deformable_groups = math.gcd(in_ch, deformable_groups)
        self.offset_conv = nn.Conv2d(in_ch, 2 * deformable_groups * kernel_size ** 2, kernel_size, 1, 1)
        self.dconv = DeformConv2d(in_ch, out_ch, kernel_size, 1, 1, bias=False)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.dconv(x, offset)
        return out

class DeformBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, blocks):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.dcblock = nn.Sequential(*[DeformConv(mid_channels, mid_channels) for _ in range(blocks)])
        self.conv_out = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.dcblock(x)
        x = self.conv_out(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv2(self.relu(self.conv1(x)))
        return x + res

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=64, blocks=30):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                  nn.LeakyReLU(0.1))
        self.res_block = nn.Sequential(*[ResidualConv(out_ch) for _ in range(blocks)])

    def forward(self, x):
        x = self.conv(x)
        return self.res_block(x)

class ConvST(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()

        self.conv_xy = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False)

        self.conv_t = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False)

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv_xy(x)
        x = self.conv_t(x)
        x = rearrange(x, 'b c t h w -> b t c h w')
        return x

class ConvSTBlock(nn.Module):
    def __init__(self, blocks, in_ch, out_ch):
        super().__init__()
        self.conv_in = nn.Conv3d(in_ch, out_ch, 3, 1, 1)
        self.block = nn.Sequential(*[ConvST(out_ch, out_ch) for _ in range(blocks)])

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv_in(x)
        x = rearrange(x, 'b c t h w -> b t c h w')
        x = self.block(x)
        return x