import torch
import torch.nn as nn
from core.modules.conv import ResidualBlock
from vsr.models.RealBasicVSR.modules.moebasic import BasicVSR

class RealBasicVSR(nn.Module):
    def __init__(self, cleaning_blocks=20, threshold=1., *args, **kwargs):
        super().__init__()
        self.cleaner = IterativeRefinement(kwargs["mid_channels"], cleaning_blocks)
        self.basicvsr = BasicVSR(*args, **kwargs)
        self.threshold = threshold

    def forward(self, lr):
        n, t, c, h, w = lr.size()
        lr = lr.reshape(-1, c, h, w)
        for _ in range(3):  # at most 3 cleaning, determined empirically
            residues = self.cleaner(lr)
            lr += residues
            if torch.mean(torch.abs(residues)) < self.threshold:
                break
        lr = lr.reshape(n, t, c, h, w)
        sr = self.basicvsr(lr)

        return sr, lr

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks):
        super().__init__()
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.resblock(x)
        return self.conv(x)
