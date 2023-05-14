import argparse
import time
import warnings

import deepspeed
import hydra
import omegaconf
import torch
import torch.distributed as dist
import wandb

from core.losses import CharbonnierLoss
from core.utils import (
    get_resources_ds,
    seed_index_everything,
    cleanup,
    build_logger,
    save_config,
    save_checkpoint_ds,
    build_model,
    create_moe_param_groups,
    build_loaders_ds,
    compute_loss,
    running_metrics,
    build_metric
)

warnings.filterwarnings('ignore')

def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--experiment',
                        type=str,
                        default="vrt_moe",
                        help='hydra experiment yaml')

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

@torch.no_grad()
def evaluate(rank, world_size, epoch, model_engine, logger, val_dl, loss_fn, metric, cfg):
    model_engine.eval()
    val_loss, val_metrics = 0, {k: 0 for k in cfg.train.metric.metrics}

    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)

        with torch.cuda.amp.autocast():
            sr, lq = model_engine(lr)
            loss = compute_loss(loss_fn, sr, hr)

        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().item() / world_size
        val_metrics = running_metrics(val_metrics, metric, sr, hr)

    save_checkpoint_ds(cfg, model_engine, logger, rank)

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_dict({k: v / len(val_dl) for k, v in val_metrics.items()}, epoch, "Val")
        logger.log_images("Val", epoch, lr, sr, hr, lq)


def run(cfg: omegaconf.DictConfig, args):
    seed_index_everything(cfg.train)
    rank, local_rank, world_size = get_resources_ds()

    # Initialize logger
    if rank == 0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        logger = build_logger(cfg)
        model_config = save_config(cfg)
    else:
        logger = None

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    if rank == 0: print('build model ...')
    model = build_model(cfg.train.model, device, local_rank, False, cfg.train.restore)

    if rank == 0: print('build engine ...')
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args, model=model, model_parameters=create_moe_param_groups(model))

    # Prepare dataset and dataloader
    if rank == 0: print('build loaders ...')
    train_dl, val_dl, epoch = build_loaders_ds(
        cfg, model_engine.train_micro_batch_size_per_gpu()
    )

    if rank == 0: print('build metrics and losses ...')
    loss_fn, metric = CharbonnierLoss(), build_metric(cfg.train.metric).to(device)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(cfg.train.max_epochs):
        model.train();
        dt = time.time()
        train_loss, train_metrics = 0.0, {k: 0 for k in cfg.train.metric.metrics}

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                sr, lq = model_engine(lr)
                loss = compute_loss(loss_fn, sr, hr, lq)

            model_engine.backward(loss)
            model_engine.step()

            train_loss += loss.detach().item()
            train_metrics = running_metrics(train_metrics, metric, sr, hr)

        if rank == 0:
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_dict({k: v / len(train_dl) for k, v in train_metrics.items()}, epoch, "Train")
            logger.log_images("Train", epoch, lr, sr, hr, lq)

            print("Starting Evaluation ...")

        evaluate(rank, world_size, epoch, model_engine,
                 logger, val_dl, loss_fn, metric, cfg)

        if rank == 0:
            dt = time.time() - dt
            print(f"Epoch {epoch} - Elapsed time --> {dt:2f}")

    if rank == 0:
        logger.close()

    return model_config

def main():
    try:
        args = add_argument()
        hydra.initialize("../conf", version_base="1.3")
        config = hydra.compose(
            config_name='default',
            overrides=[f"+experiment={args.experiment}"],
        )
        run(config, args)

    except Exception as e:
        cleanup()
        wandb.finish()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        raise e

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cleanup()

if __name__ == "__main__":
    main()
