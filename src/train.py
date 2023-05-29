import time
import warnings

import hydra
import omegaconf
import torch
import torch.distributed as dist
import wandb

from core import PROJECT_ROOT
from core.losses import CharbonnierLoss
from core.utils import (
    seed_index_everything,
    get_resources,
    compute_loss,
    running_metrics,
    setup_train,
    build_loaders,
    build_logger,
    build_metric,
    save_checkpoint,
    save_config,
    update_weights,
    cleanup
)

warnings.filterwarnings('ignore')

@torch.no_grad()
def evaluate(rank, world_size, epoch, model, optimizer, scheduler, logger, device, val_dl, loss_fn, metric, cfg):
    model.eval()
    val_loss, val_metrics = 0, {k: 0 for k in cfg.train.metric.metrics}

    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)

        with torch.cuda.amp.autocast():
            sr, lq = model(lr)
            loss = compute_loss(loss_fn, sr, hr)

        if cfg.train.ddp:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().item() / world_size
        val_metrics = running_metrics(val_metrics, metric, sr, hr)

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_dict({k: v / len(val_dl) for k, v in val_metrics.items()}, epoch, "Val")
        logger.log_images("Val", epoch, lr, sr, hr, lq)
        save_checkpoint(cfg, model, optimizer, scheduler, epoch, logger, cfg.train.ddp)

def run(cfg: omegaconf.DictConfig):
    seed_index_everything(cfg.train)
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    device = torch.device("cuda:{}".format(local_rank))

    # Initialize logger
    if rank == 0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        resume = False if cfg.train.restore is None else True
        logger = build_logger(cfg, resume)
        model_config = save_config(cfg)
    else:
        logger = None

    # Encapsulate the model on the GPU assigned to the current process
    if rank == 0: print('setup train ...')
    model, optimizer, scheduler, start_epoch = setup_train(cfg, cfg.train.model, cfg.train.optimizer,
                                                           cfg.train.scheduler, device, local_rank)

    # Mixed precision
    if rank == 0: print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # Prepare dataset and dataloader
    if rank == 0: print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    if rank == 0: print('build metrics and losses ...')
    loss_fn, metric = CharbonnierLoss(), build_metric(cfg.train.metric).to(device)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(start_epoch, cfg.train.max_epochs):
        model.train();
        dt = time.time()
        train_loss, train_metrics = 0.0, {k: 0 for k in cfg.train.metric.metrics}

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                sr, lq = model(lr)
                loss = compute_loss(loss_fn, sr, hr, lq)

            update_weights(model, loss, scaler, scheduler,
                           optimizer, num_grad_acc, gradient_clip_val, i)

            train_loss += loss.detach().item()
            train_metrics = running_metrics(train_metrics, metric, sr, hr)

        if rank == 0:
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_dict({k: v / len(train_dl) for k, v in train_metrics.items()}, epoch, "Train")
            logger.log_images("Train", epoch, lr, sr, hr, lq)

            print("Starting Evaluation ...")

        evaluate(rank, world_size, epoch, model, optimizer, scheduler,
                 logger, device, val_dl, loss_fn, metric, cfg)

        if rank == 0:
            dt = time.time() - dt
            print(f"Epoch {epoch} - Elapsed time --> {dt:2f}")

    if rank == 0:
        logger.close()

    return model_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    try:
        run(config)
    except Exception as e:
        if config.train.ddp:
            cleanup()
        wandb.finish()
        raise e

    if config.train.ddp:
        cleanup()

if __name__ == "__main__":
    main()
