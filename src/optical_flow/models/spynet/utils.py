import os
from pathlib import Path
from typing import Sequence

import hydra
import torch
from optical_flow.dataset import Dataset
from optical_flow.models import spynet
from optical_flow.transforms import *
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
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

def build_spynets(cfg, k: int, previous: Sequence[torch.nn.Module], local_rank, device):
    if cfg.train.restore is not None:
        assert not cfg.train.finetune, "Only one of restore and finetune option can be specified"
        pretrained = spynet.SpyNet.from_pretrained(cfg.train.k, cfg.train.restore)
        current_train = pretrained.units[k]

    elif cfg.train.finetune:
        assert not bool(cfg.train.restore), "Only one of restore and finetune option can be specified"
        pretrained = spynet.SpyNet.from_pretrained(cfg.train.k, None)
        current_train = pretrained.units[k]

    else:
        current_train = spynet.BasicModule()

    current_train.to(device)

    if cfg.train.ddp:
        print("Instantiating DDP Model")
        current_train = torch.nn.parallel.DistributedDataParallel(
            current_train,
            device_ids=[local_rank],
            output_device=local_rank
        )

    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(previous)
        Gk.to(device)

        if cfg.train.ddp:
            print("Instantiating DDP Pyramid")
            Gk = torch.nn.parallel.DistributedDataParallel(
                Gk,
                device_ids=[local_rank],
                output_device=local_rank
            )

        Gk.eval()

    current_train.train()
    return current_train, Gk

def update_weights_amp(loss, model, scheduler, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
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

def build_cleaner(cfg, local_rank, device):
    cleaner = hydra.utils.instantiate(cfg.train.cleaner, _recursive_=False)
    cleaner.load_state_dict(torch.load(cfg.train.cleaner_ckpt))
    cleaner.to(device)

    if cfg.train.ddp:
        print("Instantiating DDP cleaner")
        cleaner = torch.nn.parallel.DistributedDataParallel(
            cleaner,
            device_ids=[local_rank],
            output_device=local_rank
        )

    for p in cleaner.parameters():
        p.requires_grad = False

    return cleaner

def load_data(cfg, k: int):
    path = cfg.train.data.datasets.train.path

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

    train_ds = Dataset(path, "train", 0.9, augmentation=train_tfms, compression=compression)
    val_ds = Dataset(path, "val", 0.9, augmentation=val_tfms, compression=compression)

    return train_ds, val_ds

def build_dl(train_ds, val_ds, cfg):
    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_ds) if cfg.train.ddp else None
    val_sampler = DistributedSampler(dataset=val_ds) if cfg.train.ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        # persistent_workers=True,
        # pin_memory=True,
        sampler=train_sampler
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        # persistent_workers=True,
        # pin_memory=True,
        sampler=val_sampler,
        shuffle=False
    )

    return train_dl, val_dl, 0
