import torch.nn as nn
from einops import rearrange
from torch.nn.utils import spectral_norm
from torchvision.ops import DeformConv2d, deform_conv2d

class SpectralConv(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False))

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, *args, **kwargs),
                                  nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class ConvLeaky(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, *args, **kwargs),
                                  nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.conv(x)

class DeformConv(DeformConv2d):
    '''
    torch based Deformable Convolution Pack
    '''

    def __init__(self, deformable_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

class DeformBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, blocks):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.dcblock = nn.Sequential(*[
            DeformConv(deformable_groups=1, in_channels=mid_channels, out_channels=mid_channels, kernel_size=3,
                       padding=1) for _ in range(blocks)])
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

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks, steps):
        super().__init__()
        self.steps = steps
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        # n, t, c, h, w = x.size()
        # x = x.view(-1, c, h, w)
        for _ in range(self.steps):  # at most 3 cleaning, determined empirically
            residues = self.conv(self.resblock(x))
            x += residues
        return x
