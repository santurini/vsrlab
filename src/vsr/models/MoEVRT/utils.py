import hydra
import torch
from fmoe.distributed import DistributedGroupedDataParallel
from torch.nn.utils import clip_grad_norm_

def restore_model(model, path):
    state_dict = torch.load(path)['model_state_dict']
    model.load_state_dict(state_dict)
    return model

def build_model(cfg, device, ddp=False, restore_ckpt=None):
    model = hydra.utils.instantiate(cfg, _recursive_=False)
    model = model.to(device)

    if restore_ckpt is not None:
        print("restoring model state ...")
        model = restore_model(model, restore_ckpt)

    if ddp:
        print(f"setting up distributed model")
        ddp_model = DistributedGroupedDataParallel(model)
        return ddp_model

    return model

def build_optimizer(model, optim_cfg, sched_cfg, restore_ckpt=None):
    start_epoch = 0
    optimizer = hydra.utils.instantiate(optim_cfg,
                                        model.parameters(),
                                        _recursive_=False,
                                        _convert_="partial"
                                        )

    scheduler = hydra.utils.instantiate(
        sched_cfg,
        optimizer,
        _recursive_=False
    )

    if restore_ckpt is not None:
        print("restoring optimizer state ...")
        state_dict = torch.load(restore_ckpt)
        start_epoch = state_dict["epoch"] + 1
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    return optimizer, scheduler, start_epoch

def setup_train(cfg, model_cfg, optim_cfg, sched_cfg, device):
    model = build_model(model_cfg, device, cfg.train.ddp, cfg.train.restore)
    restore = None if cfg.train.finetune else cfg.train.restore

    print('restoring optimizer state from ->', restore)
    optimizer, scheduler, start_epoch = build_optimizer(model, optim_cfg, sched_cfg, restore)

    return model, optimizer, scheduler, start_epoch

def update_weights(model, loss, scaler, scheduler, optimizer, num_grad_acc, grad_clip, i):
    loss = loss / num_grad_acc
    scaler.scale(loss).backward()
    model.allreduce_gradients()

    if (i + 1) % num_grad_acc == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
