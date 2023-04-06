import torch
import torch.nn as nn
from torchvision import models

from optical_flow.modules.spynet import Spynet
from kornia.geometry.transform import resize


LAYER_WEIGHTS = {'2': 0.1, '7': 0.1, '16': 1.0, '25': 1.0, '34': 1.0}

class WL1Loss(nn.L1Loss):
    def __init__(self, weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'wl1loss'
        self.weight = weight
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        return self.loss(x, y) * self.weight

class CharbonierLoss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-12):
        super().__init__()
        self.name = 'charbonier'
        self.eps = eps
        self.weight = weight

    def forward(self, yhat, y):
        h = y.shape[-2];
        w = y.shape[-1]
        yhat = yhat.view(-1, 3, h, w)
        y = y.view(-1, 3, h, w)
        loss = torch.mean(torch.sqrt((yhat - y) ** 2 + self.eps))
        return loss * self.weight

class PerceptualVGG(nn.Module):
    def __init__(self, layer_name_list):
        super().__init__()
        self.layer_name_list = layer_name_list
        num_layers = max(map(int, self.layer_name_list)) + 1
        self.vgg_layers = models.vgg19(pretrained=True).features.requires_grad_(False)[:num_layers]

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

class PerceptualLoss(nn.Module):
    def __init__(self, weight=1, layer_weights=LAYER_WEIGHTS):
        super().__init__()
        self.name = 'perceptual'
        self.weight = weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(list(layer_weights.keys()))
        self.criterion = nn.L1Loss()

    def forward(self, yhat, y):
        h = y.shape[-2];
        w = y.shape[-1]
        yhat = yhat.view(-1, 3, h, w)
        y = y.view(-1, 3, h, w)
        x_features = self.vgg(yhat)
        gt_features = self.vgg(y.detach())
        percep_loss = 0
        for k in x_features.keys():
            percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        return percep_loss * self.weight

class AdversarialLoss(nn.Module):
    def __init__(self, weight=2e-5):
        super().__init__()
        self.name = 'adversarial'
        self.weight = weight
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, target, is_disc=False):
        target = x.new_ones(x.size()) * target
        loss = self.loss(input, target)
        return loss if is_disc else loss * self.weight

def epe_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = (target - pred).pow(2).sum().sqrt()
    return dist.mean()

class EPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @staticmethod
    def forward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return epe_loss(pred, target)

class OpticalFlowConsistency(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.spynet = Spynet().requires_grad_(False)
        self.weight = weight

    def forward(self, sr, hr):
        b, t, c, h, w = sr.shape
        img1 = sr[:, :-1, :, :, :].reshape(-1, c, hl, wl)
        img2 = sr[:, 1:, :, :, :].reshape(-1, c, hl, wl)
        flow_sr = self.spynet(img2, img1)[-1]

        img1 = hr[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        img2 = hr[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flow_hr = self.model(img2, img1)[-1]

        return epe_loss(flow_sr, flow_hr) * self.weight