import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def get_resources():

    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # from slurm
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])

    return rank, local_rank, world_size


rank, local_rank, world_size = get_resources()

num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

batch_size = 32
learning_rate = 0.1
num_epochs = 10

dist.init_process_group("nccl", rank=rank, world_size=world_size)

if rank == 0:
    print("world_size", dist.get_world_size())

device = torch.device("cuda:{}".format(local_rank))

torch.cuda.set_device(local_rank)
tr_dl, val_dl = get_loaders(cfg.loaders)

model = build_model(cfg.model, device)

ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = build_optimizer(cfg.optimizer)

criterion = build_loss(cfg.loss)

metric = build_metric(cfg.metric)

def evaluate(model, device, test_loader):
    model.eval()
    ssim = 0
    psnr = 0
    with torch.no_grad():
        for data in test_loader:
            lr, hr = data[0].to(device), data[1].to(device)
            sr = model(images)
            total += hr.size(0)
            metrics = metric(sr, hr)
            psnr += metrics['psnr']
            ssim += metrics['ssim']
    ssim = ssim / total
    psnr = psnr / total
    return ssim, psnr

epochs_time = time.time()

for epoch in range(num_epochs):

    dt = time.time()
    ddp_model.train()
    for data in train_loader:
        lr, sr = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = ddp_model.train_step(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    if rank == 0:
        psnr, ssim = evaluate(model=ddp_model, device=device, test_loader=test_loader)
        print(f"epoch {epoch} rank {rank} world_size {world_size}: psnr {ssim}, ssim {ssim}")

    dt = time.time() - dt
    if rank == 0:
        print(f"epoch {epoch} rank {rank} world_size {world_size} time {dt:2f}")

epochs_time = time.time() - epochs_time

if rank == 0:
    print(f"walltime: rank {rank} world_size {world_size} time {epochs_time:2f}")

dist.destroy_process_group()

