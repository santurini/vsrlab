import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import resize
from optical_flow.models.spynet import Spynet
from torchvision import models

LAYER_WEIGHTS = {'2': 0.1, '7': 0.1, '16': 0.8, '25': 0.9, '34': 1.0}

class WL1Loss(nn.L1Loss):
    def __init__(self, weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        return self.loss(x, y) * self.weight

class PerceptualVGG(nn.Module):
    def __init__(self, layer_name_list):
        super().__init__()
        self.layer_name_list = layer_name_list
        num_layers = max(map(int, self.layer_name_list)) + 1
        self.vgg_layers = models.vgg19(pretrained=True).features[:num_layers]

        for p in self.vgg_layers.parameters():
            p.requires_grad = False

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x
        return output

class PerceptualLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.layer_weights = LAYER_WEIGHTS
        self.vgg = PerceptualVGG(list(self.layer_weights.keys()))

    def forward(self, yhat, y):
        h = y.shape[-2];
        w = y.shape[-1]
        yhat = yhat.reshape(-1, 3, h, w)
        y = y.reshape(-1, 3, h, w)
        x_features = self.vgg(yhat)
        gt_features = self.vgg(y.detach())
        percep_loss = 0
        for k in x_features.keys():
            percep_loss += F.l1_loss(x_features[k], gt_features[k]) * self.layer_weights[k]
        return percep_loss * self.weight

class AdversarialLoss(nn.Module):
    def __init__(self, weight=2e-5):
        super().__init__()
        self.weight = weight

    def forward(self, x, target, is_disc=False):
        target = x.new_ones(x.size()) * target
        loss = F.binary_cross_entropy_with_logits(x, target)
        return loss if is_disc else loss * self.weight

def rmse_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

class OpticalFlowConsistency(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.spynet = Spynet().requires_grad_(False)
        self.weight = weight

    def forward(self, sr, hr):
        b, t, c, h, w = sr.shape
        img1 = sr[:, :-1, :, :, :].reshape(-1, c, h, w)
        img2 = sr[:, 1:, :, :, :].reshape(-1, c, h, w)
        flow_sr = self.spynet(img2, img1)[-1]

        img1 = hr[:, :-1, :, :, :].reshape(-1, c, h, w)  # remove last frame
        img2 = hr[:, 1:, :, :, :].reshape(-1, c, h, w)  # remove first frame
        flow_hr = self.spynet(img2, img1)[-1]

        return F.l1_loss(flow_sr, flow_hr) * self.weight

class LossPipeline(nn.ModuleDict):
    def __init__(self, losses, pipeline, prefix=None, postfix=None):
        super().__init__()
        self.prefix = prefix
        self.postfix = postfix
        self.pipeline = pipeline
        self.losses = losses
        self.add_losses(losses)

    def forward(self, args: dict):
        args = self.set_keys(args)
        for cfg in self.pipeline:
            loss, k = self.get_loss(args, cfg)
            args[self._set_name(k)] += loss
            args[self._set_name("loss")] += loss

        return args

    def add_losses(self, losses: dict):
        for name in sorted(losses.keys()):
            loss = losses[name]
            if not isinstance(loss, nn.Module):
                raise ValueError(
                    f"Value {loss} belonging to key {name} is not an instance of `nn.Module`"
                )

            if name in self:
                raise ValueError(f"Encountered two losses both named {name}")

            self[name] = loss

    def set_keys(self, args):
        for key in self.losses.keys():
            args[self._set_name(key)] = 0

        args[self._set_name("loss")] = 0
        return args

    def clone(self, prefix=None, postfix=None):
        ls = deepcopy(self)
        if prefix:
            ls.prefix = prefix
        if postfix:
            ls.postfix = postfix
        return ls

    def _set_name(self, base: str) -> str:
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

    def get_loss(self, args, cfg):
        for k, v in cfg.items():
            loss_fn = self[k]
            pred_key = v['x']
            gt_key = v['y']

        if "match" in pred_key:
            pred, gt = self.match_shapes(args[pred_key.removeprefix('match_')], args[gt_key])

        elif "match" in gt_key:
            gt, pred = self.match_shapes(args[gt_key.removeprefix('match_')], args[pred_key])

        else:
            pred = args[pred_key]
            gt = args[gt_key]

        return loss_fn(pred, gt), k

    @staticmethod
    def match_shapes(matching, target):
        h, w = target.shape[-2:]
        matching = resize(matching, (h, w))
        return matching, target
