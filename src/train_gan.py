import time
import warnings

import omegaconf

from core import PROJECT_ROOT
from core.losses import CharbonnierLoss
from core.utils import *

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(rank, world_size, epoch, model, logger, device, val_dl, loss_fn, metric, cfg):
    model.eval()
    val_loss, val_metrics = 0, {k: 0 for k in cfg.train.metric.metrics}

    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)

        with torch.cuda.amp.autocast():
            sr, _ = model(lr)
            loss = compute_loss(loss_fn, sr, hr)

        if cfg.train.ddp:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)

        val_loss += loss.detach().cpu().item() / world_size
        val_metrics = running_metrics(val_metrics, metric, sr, hr)

    if rank == 0:
        logger.log_dict({"LossG": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_dict({k: v / len(val_dl) for k, v in val_metrics.items()}, epoch, "Val")
        logger.log_images("Val", epoch, lr, sr, hr)
        save_checkpoint(cfg, model, logger, cfg.train.ddp)

def generator_step(model, discriminator, loss_fn, perceptual_loss, adversarial_loss, lr, hr):
    b, t, c, h, w = hr.shape

    with torch.cuda.amp.autocast():
        sr, lq = model(lr)
        pixel_loss = compute_loss(loss_fn, sr, hr, lq)
        disc_sr = discriminator(sr.reshape(-1, c, h, w))
        perceptual_g = perceptual_loss(sr, hr)

    disc_fake_loss = adversarial_loss(disc_sr, 1, False)
    loss = pixel_loss + perceptual_g + disc_fake_loss

    return sr, loss, perceptual_g, disc_fake_loss

def discriminator_step(discriminator, adversarial_loss, sr, hr):
    sr = rearrange(sr, 'b t c h w -> (b t) c h w')
    hr = rearrange(hr, 'b t c h w -> (b t) c h w')

    with torch.cuda.amp.autocast():
        disc_hr = discriminator(hr)
        disc_sr = discriminator(sr.detach())

    loss = adversarial_loss(disc_hr, 1, True) + adversarial_loss(disc_sr, 0, True)
    return loss

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
    print('build model ...')
    model = build_model(cfg.train.model, device, local_rank, cfg.train.ddp, cfg.train.restore, cfg.train.from_lightning)

    print('build discriminator ...')
    discriminator = build_model(cfg.train.discriminator, device, local_rank, cfg.train.ddp)

    # Mixed precision
    print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # Prepare dataset and dataloader
    print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    print('build optimizers and schedulers ...')
    optimizer_g, scheduler_g = build_optimizer(model,
                                               cfg.train.optimizer.generator,
                                               cfg.train.scheduler.generator
                                               )

    optimizer_d, scheduler_d = build_optimizer(discriminator,
                                               cfg.train.optimizer.discriminator,
                                               cfg.train.scheduler.discriminator
                                               )


    print('build metrics and losses ...')
    loss_fn = CharbonnierLoss()
    adversarial_loss = hydra.utils.instantiate(cfg.train.adversarial_loss, _recursive_=False)
    perceptual_loss = hydra.utils.instantiate(cfg.train.perceptual_loss, _recursive_=False).to(device)
    metric = build_metric(cfg.train.metric).to(device)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(cfg.train.max_epochs):
        model.train(); dt = time.time()
        train_losses, train_metrics = create_gan_losses_dict(), {k: 0 for k in cfg.train.metric.metrics}

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            sr, loss_g, perceptual_g, adversarial_g = generator_step(model, discriminator, loss_fn,
                                                    perceptual_loss, adversarial_loss, lr, hr)

            update_weights(model, loss_g, scaler, scheduler_g,
                           optimizer_g, num_grad_acc, gradient_clip_val, i)

            loss_d = discriminator_step(discriminator, adversarial_loss, sr, hr)

            update_weights(discriminator, loss_d, scaler, scheduler_d,
                           optimizer_d, num_grad_acc, gradient_clip_val, i)

            train_losses = running_losses(loss_g, perceptual_g, adversarial_g, loss_d, train_losses)
            train_metrics = running_metrics(train_metrics, metric, sr, hr)

        if rank == 0:
            logger.log_dict({k: v / len(train_dl) for k, v in train_losses.items()}, epoch, "Train")
            logger.log_dict({k: v / len(train_dl) for k, v in train_metrics.items()}, epoch, "Train")
            logger.log_images("Train", epoch, lr, sr, hr)

        print("Starting Evaluation ...")
        evaluate(rank, world_size, epoch, model, logger, device,
                 val_dl, loss_fn, metric, cfg)

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