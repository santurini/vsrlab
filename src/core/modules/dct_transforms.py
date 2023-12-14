import math

import numpy as np
import torch
import torch.nn as nn
from einops import *

class EncoderDCT(nn.Module):
    def __init__(self, ps=4):
        super(EncoderDCT, self).__init__()
        self.dct_conv = nn.Conv2d(3, 3 * ps * ps, ps, ps, bias=False, groups=3)
        matrix = self.generate_dct_matrix(ps)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1)
        self.dct_conv.weight.data = torch.cat([self.weight] * 3, dim=0)
        self.dct_conv.weight.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.dct_conv(x)
        x = rearrange(x, '(b t) c h w -> b t (h w) c', b=b, t=t)
        return x

    @staticmethod
    def build_filter(pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        return result * math.sqrt(2)

    def generate_dct_matrix(self, ps=8):
        matrix = np.zeros((ps, ps, ps, ps))
        for u in range(ps):
            for v in range(ps):
                for i in range(ps):
                    for j in range(ps):
                        matrix[u, v, i, j] = self.build_filter(i, u, ps) * self.build_filter(j, v, ps)
        return matrix.reshape(-1, ps, ps)

class DecoderIDCT(nn.Module):
    def __init__(self, out_ch, ps, h, w):
        super(DecoderIDCT, self).__init__()
        self.reverse_dct = nn.ConvTranspose2d(3 * ps * ps, out_ch, ps, ps, bias=False, groups=out_ch)
        matrix = self.generate_dct_matrix(ps)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1)
        self.reverse_dct.weight.data = torch.cat([self.weight] * 3, dim=0)
        self.reverse_dct.weight.requires_grad = False
        self.h = h // ps
        self.w = w // ps

    def forward(self, x):
        b, t, p, c = x.shape
        x = rearrange(x, 'b t (h w) c -> (b t) c h w', h=self.h, w=self.w)
        x = self.reverse_dct(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        return x

    @staticmethod
    def build_filter(pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        return result * math.sqrt(2)

    def generate_dct_matrix(self, ps=8):
        matrix = np.zeros((ps, ps, ps, ps))
        for u in range(ps):
            for v in range(ps):
                for i in range(ps):
                    for j in range(ps):
                        matrix[u, v, i, j] = self.build_filter(i, u, ps) * self.build_filter(j, v, ps)
        return matrix.reshape(-1, ps, ps)
