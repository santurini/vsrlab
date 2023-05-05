import logging
import os
from itertools import islice
from pathlib import Path
from typing import List, Optional, Union

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from omegaconf import DictConfig, ListConfig, OmegaConf
from kornia.geometry.transform import resize
from pytorch_lightning import seed_everything, Callback
from torch.nn import Sequential

from functools import reduce
from operator import add
from collections import Counter

CPU_DEVICE = torch.device("cpu")

pylogger = logging.getLogger(__name__)

def seed_index_everything(train_cfg: DictConfig, sampling_seed: int = 42) -> Optional[int]:
    if "seed_index" in train_cfg and train_cfg.seed_index is not None:
        seed_index = train_cfg.seed_index
        np.random.seed(sampling_seed)
        seeds = np.random.randint(np.iinfo(np.int32).max, size=max(42, seed_index + 1))
        seed = seeds[seed_index]
        seed_everything(seed)
        pylogger.info(f"Setting seed {seed} from seeds[{seed_index}]")
        return seed
    else:
        pylogger.warning("The seed has not been set! The reproducibility is not guaranteed.")
        return None

def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    return rank, local_rank

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

def save_checkpoint(cfg, model):
    save_path = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project,
        cfg.train.logger.id,
        "checkpoint"
        "last.ckpt"
    )

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_path)

def save_test_config(cfg):
    model = cfg.model_name
    version = Path(cfg.finetune).stem
    output_path = ''.join(['sr', Path(cfg.path_lr).name.partition('lr')[-1]])
    out_dir = str(Path(Path(cfg.path_lr).parent) / Path(output_path))

    save_path = '_'.join([
        out_dir,
        model,
        version
        ])

    Path(save_path).mkdir(exist_ok=True, parents=True)
    file_name = Path(save_path) / "config.yaml"

    with open(file_name, 'w') as file:
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        file.write(yaml_str)

    return save_path

def get_state_dict(path):
    return torch.load(path)['state_dict']

def get_model_state_dict(path):
    state_dict = torch.load(path)['state_dict']
    out = {k.partition('model.')[-1]: v for k, v in state_dict.items() if k.startswith('model.')}
    return out

def restore_model(model, path, local_rank):
    map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model

def build_callbacks(cfg: ListConfig) -> List[Callback]:
    callbacks = list()
    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks

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

def build_optimizer(cfg, model):
    optimizer = hydra.utils.instantiate(cfg.nn.module.optimizer,
                                        model.parameters(),
                                        _recursive_=False,
                                        _convert_="partial"
                                        )

    scheduler = build_scheduler(
        optimizer,
        cfg.nn.module.scheduler
    )

    return optimizer, scheduler

def build_transform(cfg: ListConfig) -> List[Sequential]:
    augmentation = list()
    for aug in cfg:
        pylogger.info(f"Adding augmentation <{aug['_target_'].split('.')[-1]}>")
        augmentation.append(hydra.utils.instantiate(aug, _recursive_=False))
    return Sequential(*augmentation)

def build_model(cfg, device, local_rank=None, ddp=False):
    model = hydra.utils.instantiate(cfg, _recursive_=False)
    model = model.to(device)

    if ddp:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        return ddp_model

    return model

def build_metric(cfg):
    metric = hydra.utils.instantiate(cfg, _recursive_=True, _convert_="partial")
    metrics_dict = {k: 0 for k in cfg.metrics}
    return metric, metrics_dict

def build_loaders(cfg):
    train_ds = hydra.utils.instantiate(cfg.nn.data.datasets.train, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.nn.data.datasets.val, _recursive_=False)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_ds)

    if cfg.train.num_grad_acc is not None:
        num_grad_acc = cfg.train.num_grad_acc
        batch_size = cfg.nn.data.batch_size // num_grad_acc
        steps = 0
        epoch = 0
    else:
        num_grad_acc = 1
        batch_size = cfg.nn.data.batch_size
        steps = 0
        epoch = 0

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=cfg.nn.data.num_workers,
                          prefetch_factor=cfg.nn.data.prefetch_factor
                          )

    # Test loader does not have to follow distributed sampling strategy
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=cfg.nn.data.batch_size,
                        sampler=train_sampler,
                        num_workers=cfg.nn.data.num_workers,
                        prefetch_factor=cfg.nn.data.prefetch_factor,
                        shuffle=False
                        )

    return train_dl, val_dl, num_grad_acc, steps, epoch

def compute_loss(loss_fn, loss_dict, sr, hr, lq=None):
    loss_dict["Loss"] += loss_fn(sr, hr)
    if lq is not None:
        _, _, _, h, w = lq.size()
        loss_dict["Loss"] += loss_fn(lq, resize(hr, (h, w)))
    return loss_dict
def compute_metric(metric, metrics_dict, sr, hr):
    metrics = metric(
        sr.detach().clamp(0, 1).contiguous(),
        hr.detach().contiguous()
    )
    metrics_dict = dict(reduce(add, map(Counter, [metrics, metrics_dict])))
    return metrics_dict

def update_weights(loss, scaler, scheduler, optimizer, num_grad_acc, steps):
    scaler.scale(loss / num_grad_acc).backward()
    if i + 1 % num_grad_acc == 0:
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad()
        steps += 1

    return steps

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
