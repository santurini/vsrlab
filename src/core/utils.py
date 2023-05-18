import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import deepspeed
import hydra
import numpy as np
import ptlflow
import torch
import torch.distributed as dist
from PIL import Image
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.nn import Sequential
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_tensor

CPU_DEVICE = torch.device("cpu")
pylogger = logging.getLogger(__name__)

def seed_index_everything(cfg: DictConfig, sampling_seed: int = 42) -> Optional[int]:
    if "seed_index" in cfg and cfg.seed_index is not None:
        seed_index = cfg.seed_index
        np.random.seed(sampling_seed)
        seeds = np.random.randint(np.iinfo(np.int32).max, size=max(42, seed_index + 1))
        seed = seeds[seed_index]
        seed_everything(seed)
        pylogger.info(f"Setting seed {seed} from seeds[{seed_index}]")
        return seed
    else:
        pylogger.warning("The seed has not been set! The reproducibility is not guaranteed.")
        return None

def get_resources():
    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        print("Launching with mpirun")
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

    else:
        # from torchrun
        print("Launching with torchrun")
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, local_rank, world_size

def get_resources_ds():
    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        print("Launching with mpirun")
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

        deepspeed.init_distributed(dist_backend="nccl", rank=rank, world_size=world_size)

    else:
        rank = 0
        local_rank = 0
        world_size = 1

    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def save_config(cfg):
    save_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id,
        "config.yaml"
    )
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    with open(save_path, 'w') as file:
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        file.write(yaml_str)

    return save_path

def save_checkpoint(cfg, model, logger, ddp=True):
    base_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id
    )

    save_path = os.path.join(
        base_path,
        "checkpoint",
        "last.ckpt"
    )

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    if ddp:
        torch.save(model.module.state_dict(), save_path)
        logger.save(save_path, base_path)
    else:
        torch.save(model.state_dict(), save_path)
        logger.save(save_path, base_path)

def save_checkpoint_ds(cfg, model, logger, rank):
    base_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id
    )

    save_dir = os.path.join(
        base_path,
        "checkpoint"
    )

    Path(save_dir).mkdir(exist_ok=True, parents=True)
    model.save_checkpoint(save_dir, "last", save_latest=True)
    if rank == 0:
        logger.save(f"{save_dir}/last*", base_path)

def get_state_dict(path, local_rank):
    map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    return torch.load(path, map_location=map_location)

def get_model_state_dict(path, local_rank):
    state_dict = get_state_dict(path, local_rank)
    return state_dict

def restore_model(model, path, local_rank):
    model.load_state_dict(get_model_state_dict(path, local_rank))
    return model

def build_scheduler(
        optimizer,
        scheduler: Union[ListConfig, DictConfig]
):
    if isinstance(scheduler, DictConfig):
        return hydra.utils.instantiate(
            scheduler,
            optimizer,
            _recursive_=False
        )

    if isinstance(scheduler, ListConfig):
        chained_scheduler = []
        for sched in scheduler:
            chained_scheduler.append(
                hydra.utils.instantiate(
                    sched,
                    optimizer,
                    _recursive_=False
                )
            )

        return torch.optim.lr_scheduler.ChainedScheduler(
            chained_scheduler
        )

def build_optimizer(model, optim_cfg, sched_cfg):
    pylogger.info(f"Building scheduler and optimizer")
    optimizer = hydra.utils.instantiate(optim_cfg,
                                        model.parameters(),
                                        _recursive_=False,
                                        _convert_="partial"
                                        )

    scheduler = build_scheduler(
        optimizer,
        sched_cfg
    )

    return optimizer, scheduler

def build_transform(cfg: ListConfig) -> List[Sequential]:
    augmentation = list()
    for aug in cfg:
        pylogger.info(f"Adding augmentation <{aug['_target_'].split('.')[-1]}>")
        augmentation.append(hydra.utils.instantiate(aug, _recursive_=False))
    return Sequential(*augmentation)

def build_model(cfg, device, local_rank=None, ddp=False, restore_ckpt=None):
    pylogger.info(f"Building Model")
    model = hydra.utils.instantiate(cfg, _recursive_=False)
    model = model.to(device)

    if restore_ckpt is not None:
        model = restore_model(model, restore_ckpt, local_rank)

    if ddp:
        pylogger.info(f"Setting up distributed model")
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
        return ddp_model

    return model

def build_flow(cfg):
    model = ptlflow.get_model(cfg.name, pretrained_ckpt=cfg.ckpt)
    return model

def build_metric(cfg):
    pylogger.info(f"Building Metrics")
    metric = hydra.utils.instantiate(cfg, _recursive_=True, _convert_="partial")
    return metric

def build_logger(cfg):
    logger = hydra.utils.instantiate(cfg.train.logger, _recursive_=False)
    logger.init(cfg)
    return logger

def build_dataset(cfg):
    train_ds = hydra.utils.instantiate(cfg.train.data.datasets.train, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.train.data.datasets.val, _recursive_=False)
    return train_ds, val_ds

def build_loaders(cfg):
    pylogger.info(f"Building Loaders")
    train_ds = hydra.utils.instantiate(cfg.train.data.datasets.train, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.train.data.datasets.val, _recursive_=False)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_ds) if cfg.train.ddp else None
    val_sampler = DistributedSampler(dataset=val_ds) if cfg.train.ddp else None

    if cfg.train.num_grad_acc is not None:
        num_grad_acc = cfg.train.num_grad_acc
        gradient_clip_val = cfg.train.gradient_clip_val
        batch_size = cfg.train.data.batch_size // num_grad_acc
        epoch = 0
    else:
        num_grad_acc = 1
        gradient_clip_val = cfg.train.gradient_clip_val
        batch_size = cfg.train.data.batch_size
        epoch = 0

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=cfg.train.data.num_workers,
                          prefetch_factor=cfg.train.data.prefetch_factor,
                          persistent_workers=True,
                          # pin_memory=True
                          )

    # Test loader does not have to follow distributed sampling strategy
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=cfg.train.data.batch_size,
                        sampler=val_sampler,
                        num_workers=cfg.train.data.num_workers,
                        prefetch_factor=cfg.train.data.prefetch_factor,
                        shuffle=False,
                        persistent_workers=True,
                        # pin_memory=True
                        )

    return train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch

def build_loaders_ds(cfg, batch_size):
    pylogger.info(f"Building Loaders")
    train_ds = hydra.utils.instantiate(cfg.train.data.datasets.train, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.train.data.datasets.val, _recursive_=False)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_ds)
    val_sampler = DistributedSampler(dataset=val_ds)

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=cfg.train.data.num_workers,
                          prefetch_factor=cfg.train.data.prefetch_factor,
                          persistent_workers=True,
                          # pin_memory=True
                          )

    # Test loader does not have to follow distributed sampling strategy
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=batch_size,
                        sampler=val_sampler,
                        num_workers=cfg.train.data.num_workers,
                        prefetch_factor=cfg.train.data.prefetch_factor,
                        shuffle=False,
                        persistent_workers=True,
                        # pin_memory=True
                        )

    return train_dl, val_dl, 0

def compute_loss(loss_fn, sr, hr, lq=None):
    loss = loss_fn(sr, hr)
    if lq is not None:
        _, _, c, h, w = lq.size()
        loss += loss_fn(lq, resize(hr, (h, w)))
    return loss

def compute_metric(metric, sr, hr):
    metrics = metric(
        rearrange(sr.detach().clamp(0, 1), 'b t c h w -> (b t) c h w').contiguous(),
        rearrange(hr.detach(), 'b t c h w -> (b t) c h w').contiguous()
    )
    return metrics

def running_metrics(metrics_dict, metric, sr, hr):
    metric_out = compute_metric(metric, sr, hr)
    out = {k: metrics_dict[k] + metric_out[k] for k in set(metrics_dict) & set(metric_out)}
    return out

def create_gan_losses_dict():
    return {
        "LossG": 0,
        "PerceptualG": 0,
        "AdversarialG": 0,
        "LossD": 0,
    }

def running_losses(loss_g, perceptual_g, adversarial_g, loss_d, lossess_dict):
    lossess_dict["LossG"] += loss_g.detach().cpu().item()
    lossess_dict["PerceptualG"] += perceptual_g.detach().cpu().item()
    lossess_dict["AdversarialG"] += adversarial_g.detach().cpu().item()
    lossess_dict["LossD"] += loss_d.detach().cpu().item()

    return lossess_dict

def update_weights(model, loss, scaler, scheduler, optimizer, num_grad_acc, grad_clip, i):
    loss = loss / num_grad_acc
    scaler.scale(loss).backward()

    if (i + 1) % num_grad_acc == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

def get_params(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return params

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    params = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }
    return split_params_into_different_moe_groups_for_optimizer(params)

def img2tensor(path):
    return to_tensor(Image.open(path))

def get_video(video_folder: str, pool):
    paths = list(sorted(Path(video_folder).glob('*')))
    out = torch.stack(list(pool.map(img2tensor, paths)))
    return out.unsqueeze(0)
