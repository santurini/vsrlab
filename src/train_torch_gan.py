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
    val_loss, val_metrics = 0, {k: 0 for k in cfg.nn.module.metric.metrics}

    for i, data in enumerate(val_dl):
        lr, hr = data[0].to(device), data[1].to(device)

        with torch.cuda.amp.autocast():
            sr, _ = model(lr)
            loss = compute_loss(loss_fn, sr, hr)

        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        val_loss += loss.detach().item() / world_size
        val_metrics = running_metrics(val_metrics, metric, sr, hr)

    if rank == 0:
        logger.log_dict({"Loss": val_loss / len(val_dl)}, epoch, "Val")
        logger.log_dict({k: v / len(val_dl) for k, v in val_metrics.items()}, epoch, "Val")
        logger.log_images("Val", epoch, lr, sr, hr, lq)
        save_checkpoint(cfg, model)

def generator_step(model, discriminator, loss_fn, perceptual, adversarial, lr, hr):
    b, t, c, h, w = hr.shape
    sr, lq = model(lr)
    pixel_loss = compute_loss(loss_fn, sr, hr, lq)
    perceptual_loss = perceptual(sr, hr)
    disc_sr = discriminator(sr.view(-1, c, h, w))
    disc_fake_loss = adversarial(disc_sr, 1, False)
    loss = pixel_loss + perceptual_loss + disc_fake_loss

    return loss, perceptual_loss, disc_fake_loss

def discriminator_step(discriminator, adversarial_loss, sr, hr):
    sr = rearrange(sr, 'b t c h w -> (b t) c h w')
    hr = rearrange(hr, 'b t c h w -> (b t) c h w')
    disc_hr = discriminator(hr)
    disc_true_loss = adversarial(disc_hr, 1, True)
    disc_sr = discriminator(sr.detach())
    disc_fake_loss = adversarial(disc_sr, 0, True)
    loss = disc_fake_loss + disc_true_loss

    return loss

def run(cfg: DictConfig):
    seed_index_everything(cfg.train)
    rank, local_rank, world_size = get_resources() if cfg.train.ddp else (0, 0, 1)

    # Initialize logger
    if rank == 0:
        print("Global Rank {} - Local Rank {} - Initializing Wandb".format(rank, local_rank))
        logger = build_logger(cfg.train.logger)
        model_config = save_config(cfg)
    else:
        logger = None

    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_model(cfg.nn.module.model, device, local_rank, cfg.train.ddp)

    print('build discriminator ...')
    discriminator = hydra.utils.instantiate(cfg.nn.module.discriminator, _recursive_=False)

    # Mixed precision
    print('build scaler ...')
    scaler = torch.cuda.amp.GradScaler()

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if cfg.finetune is not None:
        model = restore_model(model, cfg.finetune, local_rank)

    # Prepare dataset and dataloader
    print('build loaders ...')
    train_dl, val_dl, num_grad_acc, gradient_clip_val, epoch = build_loaders(cfg)

    print('build optimizers and schedulers ...')
    optimizer_g, scheduler_g = build_optimizer(model,
                                               cfg.nn.module.optimizer.generator,
                                               cfg.nn.module.scheduler.generator
                                               )
    optimizer_d, scheduler_d = build_optimizer(model,
                                               cfg.nn.module.optimizer.discriminator,
                                               cfg.nn.module.scheduler.discriminator
                                               )


    print('build metrics and losses ...')
    loss_fn, train_losses = CharbonnierLoss(), create_gan_losses_dict()
    adversarial_loss = hydra.utils.instantiate(cfg.nn.module.adversarial_loss, _recursive_=False)
    perceptual_loss = hydra.utils.instantiate(cfg.nn.module.perceptual_loss, _recursive_=False)
    metric, = build_metric(cfg.nn.module.metric).to(device),

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))
    for epoch in range(cfg.train.trainer.max_epochs):
        model.train();
        dt = time.time()
        train_loss, train_metrics = 0, {k: 0 for k in cfg.nn.module.metric.metrics}

        for i, data in enumerate(train_dl):
            lr, hr = data[0].to(device), data[1].to(device)

            with torch.cuda.amp.autocast():
                loss_g, perceptual_g, adversarial_g = generator_step(model, discriminator, loss_fn,
                                                        perceptual_loss, adversarial_loss, lr, hr)

            update_weights(model, loss_g, scaler, scheduler_g,
                           optimizer_g, num_grad_acc, gradient_clip_val, i)

            with torch.cuda.amp.autocast():
                loss_d = discriminator_step(discriminator, adversarial_loss, sr, hr)

            update_weights(model, loss_d, scaler, scheduler_d,
                           optimizer_d, num_grad_acc, gradient_clip_val, i)

            train_losses = running_losses(loss_g, perceptual_g, adversarial_g, loss_d, train_losses)
            train_metrics = running_metrics(train_metrics, metric, sr, hr)

        if rank == 0:
            logger.log_dict({k: v / len(train_dl) for k, v in train_losses.items()}, epoch, "Train")
            logger.log_dict({k: v / len(train_dl) for k, v in train_metrics.items()}, epoch, "Train")
            logger.log_images("Train", epoch, lr, sr, hr, lq)

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
    run(config)
    cleanup()

if __name__ == "__main__":
    main()