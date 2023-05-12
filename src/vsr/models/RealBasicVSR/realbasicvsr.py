import torch
import torch.nn as nn
import torch.nn.functional as F
from core.modules.conv import ResidualBlock
from kornia.geometry.transform import resize
from vsr.models.RealBasicVSR.modules.basicvsr import BasicVSR

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

    def train_step(self, lr, hr):
        n, t, c, h, w = lr.size()
        lq = lr
        for _ in range(3):  # at most 3 cleaning, determined empirically
            lq = lq.reshape(-1, c, h, w)
            residues = self.cleaner(lq)
            lq = (lq + residues).reshape(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.threshold:
                break

        sr = self.basicvsr(lq)

        loss = F.l1_loss(hr, sr) + F.l1_loss(resize(hr, (h, w), antialias=True), lq)

        return {
            "lr": lr,
            "lq": lq,
            "sr": sr,
            "hr": hr,
            "loss": loss
        }

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks):
        super().__init__()
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.resblock(x)
        return self.conv(x)
