import os
from typing import Sequence

import hydra
import torch
from core.utils import build_optimizer
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
    return frame1, frame2

def setup_train(cfg, k, previous, optim_cfg, sched_cfg, device, local_rank):
    current_level, trained_pyramid = build_spynets(cfg, k, previous, local_rank, device)
    restore = None if cfg.train.finetune else cfg.train.restore

    print('restoring optimizer state -->', restore)
    optimizer, scheduler, start_epoch = build_optimizer(current_level, optim_cfg, sched_cfg, restore)

    return current_level, trained_pyramid, optimizer, scheduler, start_epoch

def restore_pyramid(k, path):
    if k == 0:
        Gk = None
    else:
        Gk = spynet.SpyNet(k=k, return_levels=[-1])
        for i in range(k):
            sdict = torch.load(os.path.join(path, 'checkpoint_{}.tar'.format(k)))['model_state_dict']
            Gk.units[i].load_state_dict(sdict)

        Gk.to(device)
        Gk.eval()

    return Gk

def restore_level(k, path):
    sdict = torch.load(os.path.join(path, 'checkpoint_{}.tar'.format(k)))['model_state_dict']
    current_train = spynet.BasicModule().load_state_dict(sdict)
    return current_train

def build_spynets(cfg, k: int, previous: Sequence[torch.nn.Module], local_rank, device):
    if cfg.train.restore is not None:
        current_train = restore_spynet(k, cfg.train.restore)
        Gk = restore_pyramid(cfg, k, cfg.train.restore)

    else:
        current_train = spynet.BasicModule()
        if k == 0:
            Gk = None
        else:
            Gk = spynet.SpyNet(previous)
            Gk.to(device)
            Gk.eval()

    current_train.to(device)
    if cfg.train.ddp:
        print("Instantiating DDP Model")
        current_train = torch.nn.parallel.DistributedDataParallel(
            current_train,
            device_ids=[local_rank],
            output_device=local_rank
        )

    current_train.train()
    return current_train, Gk

def update_weights(loss, model, scheduler, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()

def save_k_checkpoint(cfg, k, model, optimizer, scheduler, epoch, logger, ddp=True):
    base_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id
    )

    save_path = os.path.join(
        base_path,
        "checkpoint_{}.tar".format(k)
    )

    model_state_dict = model.module.state_dict() if ddp else model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_path)

    logger.save(save_path, base_path)

def build_cleaner(cfg, device):
    cleaner = hydra.utils.instantiate(cfg.train.cleaner, _recursive_=False)
    cleaner.load_state_dict(torch.load(cfg.train.cleaner_ckpt))
    cleaner.to(device)

    for p in cleaner.parameters():
        p.requires_grad = False

    return cleaner

def load_data(cfg, k: int):
    path = cfg.train.data.datasets.train.path
    levels = cfg.train.k - 1

    train_tfms = Compose([
        Resize(*spynet.config.GConf(k).image_size),
        RandomRotation(17, 0.5),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5)
    ])

    compression = Compose([
        RandomVideoCompression(
            codec=['libx264'],
            crf=[34 - (levels - k) * 4],
            fps=[12]
        )]
    )

    val_tfms = Compose([
        Resize(*spynet.config.GConf(k).image_size)
    ])

    train_ds = Dataset(path, "train", 0.9, augmentation=train_tfms, compression=compression)
    val_ds = Dataset(path, "val", 0.9, augmentation=val_tfms, compression=compression)

    return train_ds, val_ds

def build_dl(train_ds, val_ds, cfg):
    train_sampler = DistributedSampler(dataset=train_ds) if cfg.train.ddp else None
    val_sampler = DistributedSampler(dataset=val_ds) if cfg.train.ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        sampler=train_sampler
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.data.batch_size,
        num_workers=cfg.train.data.num_workers,
        prefetch_factor=cfg.train.data.prefetch_factor,
        sampler=val_sampler,
        shuffle=False
    )

    return train_dl, val_dl, 0
