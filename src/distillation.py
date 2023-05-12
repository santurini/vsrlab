import time
import math
import warnings

import omegaconf
import torch.nn as nn
import torch.nn.functional as F
import wandb

from core import PROJECT_ROOT
from core.utils import *

warnings.filterwarnings('ignore')

class DistilledModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.refiner = hydra.utils.instantiate(cfg.train.refiner, _recursive_=False)
        self.student = hydra.utils.instantiate(cfg.train.model, _recursive_=False)
        self.teacher, self.io_adapter = build_flow(cfg.train.teacher)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, lr, hr):
        _, _, c, h, w = hr.size()

        with torch.no_grad():
            inputs = self.flow_inputs(hr)
            soft_labels = self.teacher(inputs)["flows"].squeeze(1)

        cleaned_inputs = self.refiner(lr)
        pixel_loss = F.l1_loss(cleaned_inputs, hr.view(-1, c, h, w))

        ref, supp = cleaned_inputs[1:], cleaned_inputs[:-1]
        pred_flows = self.student(ref, supp)
        optical_loss, flow = self.flow_loss(pred_flows, soft_labels)

        loss = optical_loss + pixel_loss
        return loss, cleaned_inputs, flow, soft_labels

    def flow_inputs(self, hr):
        device = hr.device
        hr = rearrange(hr, 'b t c h w -> (b t) h w c').cpu().numpy()
        inputs = self.io_adapter.prepare_inputs(hr)
        input_images = inputs["images"][0]
        supp, ref = input_images[:-1], input_images[1:]
        input_images = torch.stack((supp, ref), dim=1)
        inputs["images"] = input_images.to(device)
        return inputs

    def flow_loss(self, flow_preds, flow_gt):
        loss = 0.0
        _, _, h, w = flow_gt.size()
        for i, flow in enumerate(flow_preds):
            scale = 2 ** i
            rescaled_flow = self.rescale_flow(flow_gt, h, w, scale)
            loss += torch.sum((flow - rescaled_flow) ** 2, dim=1).sqrt().mean()
        return loss, flow_preds[0]

    @staticmethod
    def rescale_flow(flow, h, w, scale):
        flow = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                             align_corners=False)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
        flow[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
        flow[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
        return flow

@torch.no_grad()
def evaluate(rank, world_size, epoch, model, logger, device, val_dl, cfg):
    model.eval()
    val_loss = 0.0
    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)
        with torch.cuda.amp.autocast():
            loss, cleaned_inputs, flow, gt_flow = model(lr, hr)

        if cfg.train.ddp:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().item() / world_size

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_flow("Val", epoch, lr, cleaned_inputs, hr, flow, gt_flow)
        save_checkpoint(cfg, student, logger, cfg.train.ddp)

def run(cfg: DictConfig):
    seed_index_everything(cfg.train)
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

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
    model = DistilledModel(cfg).to(device)

    if cfg.train.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    # Mixed precision
    if rank == 0: print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # Prepare dataset and dataloader
    if rank == 0: print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    if rank == 0: print('build optimizer and scheduler ...')
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
                loss, cleaned_inputs, flow, gt_flow = model(lr, hr)

            update_weights(model, loss, scaler, scheduler,
                           optimizer, num_grad_acc, gradient_clip_val, i)

            train_loss += loss.detach().item()

        if rank == 0:
            logger.log_dict({"Loss": train_loss / len(train_dl)}, epoch, "Train")
            logger.log_flow("Train", epoch, lr, cleaned_inputs, hr, flow, gt_flow)

            print("Starting Evaluation ...")

        evaluate(rank, world_size, epoch, model, logger, device, val_dl, cfg)

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
