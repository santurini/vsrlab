from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.losses import PerceptualLoss
from einops import rearrange
from einops.layers.torch import Rearrange
from kornia.geometry.transform import resize
from optical_flow.models.raft.raft import RAFT

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels + growth_channels * 0, growth_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv3d(channels + growth_channels * 1, growth_channels, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv3d(channels + growth_channels * 2, growth_channels, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv3d(channels + growth_channels * 3, growth_channels, 3, 1, 1, bias=False)
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
    def __init__(self, nf, out_nc, sf):
        super().__init__()
        self.upconv = nn.Conv3d(nf, out_nc * sf ** 2, 3, 1, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x):
        out = self.upconv(x)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.pixel_shuffle(out)
        return out.permute(0, 2, 1, 3, 4)

class BiUpsampleBlock(nn.Module):
    def __init__(self, nf, out_nc, sf):
        super().__init__()
        self.excite_conv = nn.Conv3d(nf, 2 * nf, 3, 1, 1, bias=False)
        self.squeeze_conv = nn.Conv3d(2 * nf, nf, 3, 1, 1, bias=False)
        self.upconv = nn.Conv3d(nf, out_nc * sf ** 2, 3, 1, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x):
        out = self.excite_conv(x)
        out = self.squeeze_conv(out) + x
        out = self.upconv(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.pixel_shuffle(out)
        return out.permute(0, 2, 1, 3, 4)

class ConvGRU(nn.Module):
    def __init__(self, fea_dim=64, flow_dim=2):
        super().__init__()
        self.convz = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, bias=False)
        self.convr = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, bias=False)
        self.convq = nn.Conv3d(fea_dim + flow_dim, fea_dim, 3, 1, 1, bias=False)

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

        flow = self.optical_flow(img1, img0)
        flow = rearrange(flow, '(b t) c h w -> b c t h w', b=b, t=t-1)
        flow_0 = flow.new_zeros(b, 2, 1, h, w)
        flow = torch.cat([flow_0, flow], dim=2)

        fea = self.gru(fea, flow)

        return fea

class IRR3DBNet(nn.Module):
    def __init__(
            self,
            in_nc=3, out_nc=3, nf=32, nrb=2, nb=5, gc=64, sf=4,
            raft_small=True, raft_scale_factor=4, raft_pretrained=True,
            iterations=10, gamma=0.9
    ):
        super().__init__()
        self.sf = sf
        self.gamma = gamma
        self.iterations = iterations
        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=False)
        self.rrdbnet = nn.Sequential(*[RRDB(nf, gc, nrb) for _ in range(nb)])
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=False)
        self.upsample = UpsampleBlock(nf, out_nc, sf)
        self.update = UpdateBlock(raft_small, raft_scale_factor, raft_pretrained, nf)

    def forward(self, lr):
        b, c, t, h, w = lr.shape

        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        out = resize(lr, (self.sf*h, self.sf*w))

        for i in range(self.iterations):
            out = out.detach()

            fea = self.update(fea, out)
            res = self.upsample(fea)
            out -= res

        return out

    def train_step(self, lr, hr):
        b, c, t, h, w = lr.shape

        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        out = resize(lr, (self.sf*h, self.sf*w))

        total_loss = 0
        for i in range(self.iterations):
            out = out.detach()

            fea = self.update(fea, out)
            res = self.upsample(fea)
            out -= res

            loss = F.l1_loss(out, hr) * self.gamma ** (self.iterations - i - 1)
            total_loss += loss

        return {
            "lr": lr,
            "sr": out,
            "hr": hr,
            "loss": total_loss
        }

class RR3DBNet(nn.Module):
    def __init__(
            self,
            in_nc=3,
            out_nc=3,
            nf=32,
            nrb=2,
            nb=5,
            gc=64,
            sf=4,
            of_loss: Union[float, None] = None,
            perceptual_loss: Union[float, None] = None,
    ):
        super().__init__()
        self.sf = sf
        self.of_loss = of_loss
        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=False)
        self.rrdbnet = nn.Sequential(*[RRDB(nf, gc, nrb) for _ in range(nb)])
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=False)
        self.upsample = BiUpsampleBlock(nf, out_nc, sf)

        if of_loss:
            self.rearrange = Rearrange('b c t h w -> (b t) c h w')
            self.raft = RAFT(small=True, scale_factor=8, pretrained=True)
            for param in self.raft.parameters():
                param.requires_grad = False

        if perceptual_loss:
            self.perceptual = PerceptualLoss(perceptual_loss)
            for param in self.perceptual.parameters():
                param.requires_grad = False
            self.perceptual_loss = True

    def forward(self, lr):

        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        sr = self.upsample(fea)

        return sr

    def train_step(self, lr, hr):

        fea = self.conv_first(lr)
        trunk = self.trunk_conv(self.rrdbnet(fea))
        fea = fea + trunk

        sr = self.upsample(fea)

        loss = F.l1_loss(sr, hr)

        out = {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "loss": loss
        }

        if self.of_loss:
            of_loss = F.l1_loss(
                self.raft(
                    self.rearrange(sr[:, :, :-1, :, :]).detach(),
                    self.rearrange(sr[:, :, 1:, :, :]).detach()
                ),
                self.raft(
                    self.rearrange(hr[:, :, :-1, :, :]).detach(),
                    self.rearrange(hr[:, :, 1:, :, :]).detach()
                )
            ) * self.of_loss

            out["of_loss"] = of_loss
            out["loss"] += of_loss

        return out

