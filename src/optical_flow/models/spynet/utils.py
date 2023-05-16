import os
from pathlib import Path
from typing import Sequence

import hydra
import ptlflow
import torch
from einops import rearrange
from optical_flow.dataset import Dataset
from optical_flow.models import spynet
from optical_flow.transforms import *
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

@torch.no_grad()
def get_frames(lr, cleaner, size):
    cleaned_inputs = resize(cleaner(lr), size=size)
    frame1, frame2 = cleaned_inputs[:-1], cleaned_inputs[1:]
    return (frame1, frame2)

@torch.no_grad()
def clean_frames(cleaner, frame1, frame2):
    frame1, frame2 = cleaner(
        torch.concat([frame1, frame2])
    ).split(len(frame1))
    return (frame1, frame2)

@torch.no_grad()
def get_flow(hr, teacher, size):
    hr = rearrange(hr, 'b t c h w -> (b t) c h w')
    supp, ref = hr[:-1], hr[1:]
    input_images = torch.stack((supp, ref), dim=1)
    inputs = {"images": input_images}
    soft_labels = teacher(inputs)["flows"].squeeze(1)
    soft_labels = resize(soft_labels, size=size, antialias=True)
    inputs = resize(inputs["images"][0], size=size, antialias=True)
    return soft_labels, inputs

def build_spynets(cfg, k: int, previous: Sequence[torch.nn.Module], device):
    # pretrained = spynet.SpyNet.from_pretrained(cfg.train.k)
    # current_train = pretrained.units[k]

    current_train = spynet.BasicModule()
    current_train.to(device)
    current_train.train()

    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(previous)
        Gk.to(device)
        Gk.eval()

    return current_train, Gk

def update_weights_amp(loss, scheduler, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()

def update_weights(loss, scheduler, optimizer):
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

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

def build_teacher(cfg, device):
    model = ptlflow.get_model(cfg.name, pretrained_ckpt=cfg.ckpt)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)

def build_cleaner(cfg, device):
    cleaner = hydra.utils.instantiate(cfg.train.cleaner, _recursive_=False)
    cleaner.load_state_dict(torch.load(cfg.train.cleaner_ckpt))
    for p in cleaner.parameters():
        p.requires_grad = False
    return cleaner.to(device)

def load_data(cfg, k: int):
    path = cfg.train.data.datasets.train.path
    size = cfg.train.data.datasets.train.size

    train_tfms = Compose([
        Resize(*spynet.config.GConf(k).image_size),
        RandomRotation(17, 0.5),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5)
    ])

    compression = Compose([
        RandomVideoCompression(
            codec=['libx264'],
            crf=[38],
            fps=[30]
        ),
        Normalize(
            mean=[.485, .406, .456],
            std=[.229, .225, .224]
        )]
    )

    val_tfms = Compose([
        Resize(*spynet.config.GConf(k).image_size)
    ])

    train_ds = Dataset(path, "train", size, augmentation=train_tfms, compression=compression)
    val_ds = Dataset(path, "val", 1 - size, augmentation=val_tfms, compression=compression)

    return train_ds, val_ds

def build_dl(train_ds, val_ds, cfg):
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        # collate_fn=collate_fn
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        # collate_fn=collate_fn
    )

    return train_dl, val_dl, 0

def collate_fn(batch):
    frames, flow = zip(*batch)
    frame1, frame2 = zip(*frames)
    return (torch.stack(frame1), torch.stack(frame2)), torch.stack(flow)
