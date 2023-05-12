import time
import warnings

import torch
import torch.nn as nn

import omegaconf
import wandb
from einops import rearrange

from core import PROJECT_ROOT
from core.modules.conv import ResidualBlock
from core.utils import *

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

class IterativeRefinement(nn.Module):
    def __init__(self, mid_ch, blocks, steps):
        super().__init__()
        self.steps = steps
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        n, t, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        for _ in range(self.steps):  # at most 3 cleaning, determined empirically
            residues = self.conv(self.resblock(x))
            x += residues
        return x

def flow_loss(flow_preds, flow_gt):
    loss = 0.0
    for flow in flow_preds:
        _, _, h, w = flow.size()
        loss += torch.sum((flow - resize(flow_gt, size=(h,w))) ** 2, dim=1).sqrt().mean()
    return loss, flow

def distillation(teacher, student, refiner, lr, hr):
    with torch.no_grad():
        ref, supp = rearrange(hr[:, :-1, ...], 'b t c h w -> (b t) c h w'),\
            rearrange(hr[:, 1:, ...], 'b t c h w -> (b t) c h w')
        cleaned_inputs = refiner(lr) # -> b*t c h w
        soft_labels = teacher((supp, ref))
    ref, supp = cleaned_inputs[1:], cleaned_inputs[:-1]
    pred_flows = student(ref, supp)
    loss, flow = flow_loss(pred_flows, soft_labels)
    return loss, flow, soft_labels

@torch.no_grad()
def evaluate(rank, world_size, epoch, teacher, student, refiner, logger, device, val_dl, cfg):
    model.eval()
    val_loss = 0.0
    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)
        with torch.cuda.amp.autocast():
            loss, flow, gt_flow = distillation(teacher, student, refiner, lr, hr)

        if cfg.train.ddp:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().item() / world_size

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_flow("Val", epoch, lr, flow, gt_flow)
        save_checkpoint(cfg, student, logger, cfg.train.ddp)

def run(cfg: DictConfig):
    seed_index_everything(cfg.train)
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    # Initialize logger
    if rank==0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        logger = build_logger(cfg)
        model_config = save_config(cfg)
    else:
        logger = None

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    if rank==0: print('build model ...')
    model = build_model(cfg.train.model, device, local_rank, cfg.train.ddp, cfg.train.restore)

    # Mixed precision
    if rank==0: print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # Prepare dataset and dataloader
    if rank==0: print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    if rank==0: print('build optimizer and scheduler ...')
    optimizer, scheduler = build_optimizer(model, cfg.train.optimizer, cfg.train.scheduler)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(cfg.train.max_epochs):
        model.train();
        dt = time.time()
        train_loss = 0.0

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                loss, flow, gt_flow = distillation(teacher, student, refiner, lr, hr)

            update_weights(student, loss, scaler, scheduler,
                           optimizer, num_grad_acc, gradient_clip_val, i)

            train_loss += loss.detach().item()

        if rank == 0:
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_flow("Val", epoch, lr, flow, gt_flow)

            print("Starting Evaluation ...")

        evaluate(rank, world_size, epoch, teacher, student, refiner,
                    logger, device, val_dl, cfg)

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
