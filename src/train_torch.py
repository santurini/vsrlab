import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import random
import numpy as np
import time
import torch.distributed as dist

import hydra
import omegaconf
from omegaconf import DictConfig
import wandb

from core import PROJECT_ROOT
from core.utils import build_scheduler, save_config, save_checkpoint, seed_index_everything
from core.losses import CharbonnierLoss

import warnings

warnings.filterwarnings('ignore')

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def log_images(out, stage, epoch):
    lr = out["lr"][0, -1, :, :, :].detach()
    hr = out["hr"][0, -1, :, :, :].detach()
    sr = out["sr"][0, -1, :, :, :].detach().clamp(0, 1)

    grid = make_grid([lr, sr, hr], nrow=3, ncol=1)
    wandb.log({f'Prediction {stage}': [wandb.Image(grid, caption=f'Stage {stage}, Epoch {epoch}')]})

def evaluate(model, device, test_loader, criterion):
    model.eval()
    loss = 0

    with torch.no_grad():
        for data in test_loader:
            lr, hr = data[0].to(device), data[1].to(device)
            sr, _ = model(lr)
            loss += criterion(sr, hr)

    loss = loss / len(test_loader)

    return {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "loss": loss
        }

def run(cfg: DictConfig):
    save_config(cfg)
    seed_index_everything(cfg.train)

    model_filepath = os.path.join(
        cfg.train.logger.save_dir,
        cfg.train.logger.project
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Initialize logger
    wandb.init(
        dir = cfg.train.logger.save_dir,
        project = cfg.train.logger.project,
        id = cfg.train.logger.id,
        name = cfg.train.logger.name
    )

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    model = hydra.utils.instantiate(cfg.nn.module.model, _recursive_=False)
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    device = torch.device("cuda:{}".format(local_rank))

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if cfg.checkpoint is not None:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(cfg.checkpoint, map_location=map_location))

    # Prepare dataset and dataloader
    train_ds = hydra.utils.instantiate(cfg.nn.data.datasets.train, _recursive_=False)
    val_ds = hydra.utils.instantiate(cfg.nn.data.datasets.val, _recursive_=False)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_ds)

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=2, #cfg.nn.data.batch_size,
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

    optimizer = hydra.utils.instantiate(cfg.nn.module.optimizer,
                                        ddp_model.parameters(),
                                        _recursive_=False,
                                        _convert_="partial"
                                        )

    scheduler: Optional[Any] = build_scheduler(
        optimizer,
        cfg.nn.module.scheduler
    )

    loss_fn = CharbonnierLoss()

    # Loop over the dataset multiple times
    for epoch in range(2):
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if local_rank == 0:
                step_out = evaluate(model=ddp_model, device=device, test_loader=val_dl, criterion=loss_fn)
                save_checkpoint(cfg, ddp_model)
                log_images(step_out, "Dio", epoch)
                wandb.log({"Loss/Val": step_out["loss"]})

                print("-" * 75)
                print("Epoch: {}, Validation Loss: {}".format(epoch, step_out["loss"]))
                print("-" * 75)

        ddp_model.train()
        for data in train_dl:
            lr, hr = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                sr, lq = ddp_model(lr)
                loss = loss_fn(sr, hr) + loss_fn(lq, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()

            # Updates the scale for next iteration.
            scaler.update()
            wandb.log({"Loss/Train": loss})

        log_images({"sr":sr, "lr":lr, "hr":hr}, "Train", epoch)
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

    wandb.finish()

    return model_filepath

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)
    cleanup()

if __name__ == "__main__":
    main()
