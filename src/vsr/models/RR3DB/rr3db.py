import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import resize
from einops import rearrange

from optical_flow.models.raft.raft import RAFT

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels + growth_channels * 0, growth_channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv2 = nn.Conv3d(channels + growth_channels * 1, growth_channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv3 = nn.Conv3d(channels + growth_channels * 2, growth_channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv4 = nn.Conv3d(channels + growth_channels * 3, growth_channels, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv5 = nn.Conv3d(channels + growth_channels * 4, channels, 3, 1, 1, bias=False)
        self.selu = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.selu(self.conv1(x))
        x2 = self.selu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.selu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.selu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5

class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int, blocks: int):
        super().__init__()
        self.rrdb = nn.Sequential(*[
            ResidualDenseBlock(channels, growth_channels)
            for _ in range(blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rrdb(x)
        return out + x

class UpsampleBlock(nn.Module):
    def __init__(self, nf, out_nc):
        super().__init__()
        self.upconv1 = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.upconv2 = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.HRconv = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, padding_mode='reflect', bias=False)
        self.selu = nn.SELU()

    def forward(self, x):
        _, _, _, h, w = x.shape
        x = self.selu(self.upconv1(resize(x, (2 * h, 2 * w))))
        _, _, _, h, w = x.shape
        x = self.selu(self.upconv2(resize(x, (2 * h, 2 * w))))
        out = self.conv_last(self.selu(self.HRconv(x)))
        return out

class ConvGRU(nn.Module):
    def __init__(self, fea_dim=64, flow_dim=2):
        super().__init__()
        self.convz = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, padding_mode='reflect', bias=False)
        self.convr = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, padding_mode='reflect', bias=False)
        self.convq = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, padding_mode='reflect', bias=False)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h

class UpdateBlock(nn.Module):
    def __init__(self, small, scale_factor, pretrained, nf):
        super().__init__()
        self.optical_flow = RAFT(small, scale_factor, pretrained).requires_grad_(False)
        self.gru = ConvGRU(fea_dim=nf, flow_dim=2)

    def forward(self, fea, img):
        b, c, t, h, w = fea.shape
        img0 = rearrange(img[:, :, :-1, :, :], 'b c t h w -> (b t) c h w')
        img1 = rearrange(img[:, :, 1:, :, :], 'b c t h w -> (b t) c h w')

        flow = self.optical_flow(img0, img1)
        flow = rearrange(flow, '(b t) c h w -> b c t h w', b=b, t=t-1)
        flow_0 = flow.new_zeros(b, 2, 1, h, w)
        flow = torch.cat([flow_0, flow], dim=2)

        fea = self.gru(fea, flow)

        return fea

class RR3DBNet(nn.Module):
    def __init__(
            self,
            in_nc=3, out_nc=3, nf=32, nrb=2, nb=5, gc=64,
            raft_small=True, raft_scale_factor=4, raft_pretrained=True,
            iterations=10, gamma=0.9
    ):
        super().__init__()
        self.gamma = gamma
        self.iterations = iterations
        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.rrdbnet = nn.Sequential(*[RRDB(nf, gc, nrb) for _ in range(nb)])
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, padding_mode='reflect', bias=False)
        self.upsample = UpsampleBlock(nf, out_nc)
        self.update = UpdateBlock(raft_small, raft_scale_factor, raft_pretrained, nf)

    def forward(self, lr, hr, test=True):
        if test:
            return self.test_step(lr, hr)
        else:
            return self.train_step(lr, hr)

    def train_step(self, lr, hr):
        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        out = self.upsample(fea)

        total_loss = 0
        for i in range(self.iterations):
            fea = self.update(fea, out)
            res = self.upsample(fea)
            out = out + res

            loss = F.l1_loss(out, hr) * self.gamma ** (self.iterations - i)
            loss.backward(retain_graph=True)

            total_loss += loss

        return out, total_loss

    def test_step(self, lr, hr):
        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        out = self.upsample(fea)

        for i in range(self.iterations):
            fea = self.update(fea, out)
            res = self.upsample(fea)
            out = out + res

        loss = F.l1_loss(out, hr)

        return out, loss
