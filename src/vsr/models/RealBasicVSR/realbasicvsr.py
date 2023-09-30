import torch.nn as nn
from vsrlab.core.modules.conv import ResidualBlock
from vsrlab.vsr.models.RealBasicVSR.modules.basicvsr import BasicVSR

class RealBasicVSR(nn.Module):
    def __init__(self, cleaning_blocks=20, *args, **kwargs):
        super().__init__()
        self.cleaner = IterativeRefinement(kwargs["mid_channels"], cleaning_blocks)
        self.basicvsr = BasicVSR(*args, **kwargs)

    def forward(self, lr):
        lr = self.cleaner(lr)
        sr = self.basicvsr(lr)

        return sr, lr

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks, steps=3):
        super().__init__()
        self.steps = steps
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        n, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        for _ in range(self.steps):  # at most 3 cleaning, determined empirically
            residues = self.conv(self.resblock(x))
            x += residues
        return x.view(n, t, c, h, w)
