import torch.nn as nn
from core.modules.conv import ConvST

class PixelShufflePack(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor):
        super().__init__()
        self.upconv = nn.Conv2d(in_ch, out_ch * upscale_factor * upscale_factor, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.upconv(x)
        return self.lrelu(self.pixel_shuffle(x))

class PixelShufflePack3D(nn.Module):
    def __init__(self, in_ch, out_ch, upscale):
        super().__init__()
        self.mapping = ConvST(in_ch, out_ch * upscale ** 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.mapping(x)
        out = self.pixel_shuffle(x)
        return out