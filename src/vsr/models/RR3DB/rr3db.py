import torch
import torch.nn as nn
from kornia.geometry.transform import resize

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h

class UpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super().__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, fea, enc, img):
        flow = self.optical_flow(fea, img)
        flow_fea = flow_warp(fea, flow)
        inp = torch.cat([img, flow_fea], dim=1)
        fea = self.gru(fea, flow_fea)
        res_img = self.conv_last(self.lrelu(self.HRconv(fea)))

        return fea, res_img

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels + growth_channels * 0, growth_channels, 3, 1, 1, padding_mode='reflect',
                               bias=False)
        self.conv2 = nn.Conv3d(channels + growth_channels * 1, growth_channels, 3, 1, 1, padding_mode='reflect',
                               bias=False)
        self.conv3 = nn.Conv3d(channels + growth_channels * 2, growth_channels, 3, 1, 1, padding_mode='reflect',
                               bias=False)
        self.conv4 = nn.Conv3d(channels + growth_channels * 3, growth_channels, 3, 1, 1, padding_mode='reflect',
                               bias=False)
        self.conv5 = nn.Conv3d(channels + growth_channels * 4, channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5

class ResidualResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out + x

class UpsampleBlock(nn.Module):
    def __init__(self):
        self.upsample = nn.Sequential(
            Resize((h * 2, w * 2)),
            nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.LeakyReLU(0.2),
            Resize((h * 4, w * 4)),
            nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv3d(nf, 3, 3, 1, 1, padding_mode='reflect', bias=False),
        )

class RR3DBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        self.sf = sf
        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        #### upsampling
        self.upconv1 = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.upconv2 = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.HRconv = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, padding_mode='reflect', bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        _, _, _, h, w = x.shape

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        out = self.upsample(fea)

        for _ in range(iterations):

            out = self.update_block(fea, out)
            out = x_up + out

        return out
