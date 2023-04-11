from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 + x

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