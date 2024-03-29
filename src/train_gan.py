import time
import warnings

import hydra
import omegaconf
import torch
import wandb
from einops import rearrange
from vsrlab.core import PROJECT_ROOT
from vsrlab.core.losses import CharbonnierLoss
from vsrlab.core.utils import (
    seed_index_everything,
    get_resources,
    compute_loss,
    running_metrics,
    create_gan_losses_dict,
    running_losses,
    setup_train,
    build_model,
    build_optimizer,
    build_loaders,
    build_logger,
    build_metric,
    save_config,
    update_weights,
    cleanup
)
from vsrlab.train import evaluate

warnings.filterwarnings('ignore')

def dummy_loss(x, y):
    return torch.tensor(0, dtype=torch.float32, requires_grad=True)

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
    model, optimizer_g, scheduler_g, start_epoch = setup_train(cfg, cfg.train.model, cfg.train.optimizer.generator,
                                                               cfg.train.scheduler.generator, device, local_rank)

    if rank == 0: print('build discriminator ...')
    discriminator = build_model(cfg.train.discriminator, device, local_rank, cfg.train.ddp)
    optimizer_d, scheduler_d, _ = build_optimizer(discriminator,
                                                  cfg.train.optimizer.discriminator,
                                                  cfg.train.scheduler.discriminator
                                                  )

    # Mixed precision
    if rank == 0: print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # Prepare dataset and dataloader
    if rank == 0: print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    if rank == 0: print('build metrics and losses ...')
    loss_fn = CharbonnierLoss()
    adversarial_loss = hydra.utils.instantiate(cfg.train.adversarial_loss, _recursive_=False)
    perceptual_loss = dummy_loss if cfg.train.perceptual_loss is None else hydra.utils.instantiate(
        cfg.train.perceptual_loss, _recursive_=False).to(device)
    metric = build_metric(cfg.train.metric).to(device)

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(cfg.train.max_epochs):
        model.train();
        dt = time.time()
        train_losses, train_metrics = create_gan_losses_dict(), {k: 0 for k in cfg.train.metric.metrics}

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            sr, loss_g, perceptual_g, adversarial_g = generator_step(model, discriminator, loss_fn,
                                                                     perceptual_loss, adversarial_loss, lr, hr)

            if epoch > cfg.train.freeze_epochs:
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

        evaluate(rank, world_size, epoch, model, optimizer_g, scheduler_g,
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
