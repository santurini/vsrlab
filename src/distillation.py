import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import omegaconf
import wandb
from einops import rearrange

from core import PROJECT_ROOT
from core.utils import *

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

def flow_loss(flow_preds, flow_gt):
    loss = 0.0
    for i, flow in enumerate(flow_preds):
        _, _, h, w = flow.size()
        scale = 2 ** (5 - i)
        rescaled_flow = rescale_flow(flow_gt, h, w, scale)
        loss += torch.sum((flow - rescaled_flow) ** 2, dim=1).sqrt().mean()
    return loss, flow

def rescale_flow(flow, h, w, scale):
    flow = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                             align_corners=False)
    w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
    h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
    flow[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
    flow[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
    return flow

def flow_inputs(io_adapter, hr):
    hr = rearrange(hr, 'b t c h w -> (b t) h w c').numpy()
    inputs = io_adapter.prepare_inputs(hr)
    input_images = inputs["images"][0]
    supp, ref = input_images[:-1], input_images[1:]
    input_images = torch.stack((supp, ref), dim=1)
    inputs["images"] = input_images
    return inputs

def distillation(teacher, student, refiner, io_adapter, lr, hr):
    _, _, c, h, w = hr.size()

    with torch.no_grad():
        inputs = flow_inputs(io_adapter, hr)
        soft_labels = teacher(inputs)["flows"].squeeze(1)

    cleaned_inputs = refiner(lr)
    pixel_loss = F.l1_loss(cleaned_inputs, hr.view(-1, c, h, w))

    ref, supp = cleaned_inputs[1:], cleaned_inputs[:-1]
    pred_flows = student(ref, supp)
    optical_loss, flow = flow_loss(pred_flows, soft_labels)

    loss = optical_loss + pixel_loss
    return loss, cleaned_input, flow, soft_labels

@torch.no_grad()
def evaluate(rank, world_size, epoch, teacher, student, refiner,
             io_adapter, logger, device, val_dl, cfg):
    model.eval()
    val_loss = 0.0
    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)
        with torch.cuda.amp.autocast():
            loss, cleaned_input, flow, gt_flow = distillation(teacher, student, refiner,
                                                              io_adapter, lr, hr)

        if cfg.train.ddp:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().item() / world_size

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_flow("Val", epoch, lr, cleaned_input, hr, flow, gt_flow)
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
    teacher, io_adapter = build_flow(cfg.train.teacher, device, local_rank, cfg.train.ddp)
    student = build_model(cfg.train.model, device, local_rank, cfg.train.ddp)
    refiner = build_model(cfg.train.refiner, device, local_rank, cfg.train.ddp)

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
                loss, flow, gt_flow = distillation(teacher, student, refiner, io_adapter, lr, hr)

            update_weights(student, loss, scaler, scheduler,
                           optimizer, num_grad_acc, gradient_clip_val, i)

            train_loss += loss.detach().item()

        if rank == 0:
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_flow("Val", epoch, lr, flow, gt_flow)

            print("Starting Evaluation ...")

        evaluate(rank, world_size, epoch, teacher, student, refiner,
                    io_adapter, logger, device, val_dl, cfg)

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
