import hydra
from core.utils import restore_model
from fmoe.distributed import DistributedGroupedDataParallel

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

def build_model(cfg, device, local_rank=None, ddp=False, restore_ckpt=None):
    model = hydra.utils.instantiate(cfg, _recursive_=False)
    model = model.to(device)

    if restore_ckpt is not None:
        model = restore_model(model, restore_ckpt, local_rank)

    if ddp:
        ddp_model = DistributedGroupedDataParallel(model)
        return ddp_model

    return model
