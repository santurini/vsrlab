import os
from pathlib import Path
from typing import Sequence

import torch
from einops import rearrange
from kornia.augmentation import Normalize
from optical_flow.models import spynet
from torchvision.transforms.functional import resize

normalizer = Normalize(mean=[.485, .406, .456],
                      std= [.229, .225, .224])

def get_frames(lr, cleaner, size):
    cleaned_inputs = resize(cleaner(lr), size=size)
    ref, supp = normalizer(cleaned_inputs[1:]), \
        normalizer(cleaned_inputs[:-1])
    return (ref, supp)

def get_flow(hr, teacher, size):
    hr = rearrange(hr, 'b t c h w -> (b t) c h w')
    supp, ref = hr[:-1], hr[1:]
    input_images = torch.stack((supp, ref), dim=1)
    inputs = {"images": input_images}
    soft_labels = teacher(inputs)["flows"].squeeze(1)
    soft_labels = resize(soft_labels, size=size, antialias=True)
    return soft_labels, inputs["images"]

def build_spynets(cfg, k: int, previous: Sequence[torch.nn.Module], device):
    pretrained = spynet.SpyNet.from_pretrained(cfg.train.k)
    current_train = pretrained.units[k]

    current_train.to(device)
    current_train.train()

    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(previous)
        Gk.to(device)
        Gk.eval()

    return current_train, Gk

def update_weights(loss, scheduler, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

def save_k_checkpoint(cfg, k, model, logger, ddp=True):
    base_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id
    )

    save_path = os.path.join(
        base_path,
        "checkpoint",
        f"{k}.ckpt"
    )

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    if ddp:
        torch.save(model.module.state_dict(), save_path)
        logger.save(save_path, base_path)
    else:
        torch.save(model.state_dict(), save_path)
        logger.save(save_path, base_path)

