import logging
import os
from pathlib import Path
from typing import List, Optional

import deepspeed
import hydra
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

def get_resources_ds():
    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        print("Launching with mpirun")
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

    deepspeed.init_distributed(dist_backend="nccl", rank=rank, world_size=world_size)

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

def save_checkpoint_ds(cfg, model, logger):
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
    logger.save(f"{save_dir}/*last*", base_path)

def get_state_dict(path, local_rank):
    map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    return torch.load(path, map_location=map_location)

def get_model_state_dict(path, local_rank):
    state_dict = get_state_dict(path, local_rank)
    return state_dict

def restore_model(model, path, local_rank):
    model.load_state_dict(get_model_state_dict(path, local_rank))
    return model

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

def build_logger(cfg):
    logger = hydra.utils.instantiate(cfg.train.logger, _recursive_=False)
    logger.init(cfg)
    return logger

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

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    params = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }
    return split_params_into_different_moe_groups_for_optimizer(params)
