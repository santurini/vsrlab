import torch.nn as nn
from torch.nn.utils import spectral_norm

class UNetDiscriminator(nn.Module):
    def __init__(self, in_ch=3, mid_ch=64):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.conv_1 = SpectralConv(mid_ch, mid_ch * 2, 4, 2, 1)
        self.conv_2 = SpectralConv(mid_ch * 2, mid_ch * 4, 4, 2, 1)
        self.conv_3 = SpectralConv(mid_ch * 4, mid_ch * 8, 4, 2, 1)
        self.conv_4 = SpectralConv(mid_ch * 8, mid_ch * 4, 3, 1, 1)
        self.conv_5 = SpectralConv(mid_ch * 4, mid_ch * 2, 3, 1, 1)
        self.conv_6 = SpectralConv(mid_ch * 2, mid_ch, 3, 1, 1)
        self.conv_7 = SpectralConv(mid_ch, mid_ch, 3, 1, 1)
        self.conv_8 = SpectralConv(mid_ch, mid_ch, 3, 1, 1)
        self.conv_9 = nn.Conv2d(mid_ch, 1, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, img):
        feat_0 = self.lrelu(self.conv_0(img))
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))
        feat_3 = self.upsample(feat_3)
        feat_4 = self.upsample(self.lrelu(self.conv_4(feat_3)) + feat_2)
        feat_5 = self.upsample(self.lrelu(self.conv_5(feat_4)) + feat_1)
        feat_6 = self.lrelu(self.conv_6(feat_5)) + feat_0
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))
        return self.conv_9(out)

class SpectralConv(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return spectral_norm(x)
