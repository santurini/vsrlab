import torch
import torch.nn as nn
from core.modules.conv import ResidualBlock
from vsr.models.RealBasicVSR.modules.basicvsr import BasicVSR

class RealBasicVSR(nn.Module):
    def __init__(self, cleaning_blocks=20, threshold=1., *args, **kwargs):
        super().__init__()
        self.name = 'Real Basic VSR'
        self.cleaner = CleaningModule(kwargs["mid_channels"], cleaning_blocks)
        self.basicvsr = BasicVSR(*args, **kwargs)
        self.threshold = threshold

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        for _ in range(3):  # at most 3 cleaning, determined empirically
            lqs = lqs.view(-1, c, h, w)
            residues = self.cleaner(lqs)
            lqs = (lqs + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.threshold:
                break
        sr, flow_f, flow_b = self.basicvsr(lqs)
        return sr, lqs, flow_f, flow_b

class CleaningModule(nn.Module):
    def __init__(self, mid_ch, blocks):
        super().__init__()
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.resblock(x)
        return self.conv(x)
