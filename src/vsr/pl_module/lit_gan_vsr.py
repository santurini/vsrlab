

class LitRealGanVSR(LitRealVSR):
    def __init__(self,
                 discriminator: DictConfig,
                 adversarial_loss: DictConfig,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.discriminator = hydra.utils.instantiate(discriminator, _recursive_=False)

        if self.hparams.perceptual_loss:
            self.perceptual_loss: Optional[Any] = hydra.utils.instantiate(self.hparams.perceptual_loss,
                                                                          _recursive_=False)

        self.adversarial_loss = hydra.utils.instantiate(adversarial_loss, _recursive_=False)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            return self.generator_step(batch, batch_idx, optimizer_idx)
        elif optimizer_idx == 1:
            return self.discriminator_step(batch, batch_idx, optimizer_idx)

    def generator_step(self, batch):
        lr, hr = batch
        step_out = self.step(lr, hr)
        self.discriminator.requires_grad_(False)
        perceptual_loss = self.perceptual(step_out["sr"], hr)
        disc_sr = self.discriminator(step_out["sr"])
        disc_fake_loss = self.adversarial(disc_sr, 1, False)
        loss = step_out['loss'] + perceptual_loss + disc_fake_loss

        self.log_dict(
            {"loss/train/generator": step_out['loss'].cpu().detach(),
             "loss/train/generator_pixel": loss.cpu().detach(),
             "loss/train/generator_perceptual": perceptual_loss.cpu().detach(),
             "loss/train/generator_fake": disc_fake_loss.cpu().detach(), },
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr, 'b t c h w -> (b t) c h w')
            ),
        )

        return step_out

    def discriminator_step(self, batch):
        lr, hr = batch
        step_out = self.step(lr)
        self.discriminator.requires_grad_(True)
        disc_hr = self.discriminator(hr)
        disc_true_loss = self.adversarial(disc_hr, 1, True)
        disc_sr = self.discriminator(step_out["sr"].detach())
        disc_fake_loss = self.adversarial(disc_sr, 0, True)
        loss = disc_fake_loss + disc_true_loss

        self.log_dict(
            {"loss/train/discriminator": loss.cpu().detach(),
             "loss/train/discriminator_fake": disc_fake_loss.cpu().detach(),
             "loss/train/discriminator_true": disc_true_loss.cpu().detach()},
        )

        return step_out

    def configure_optimizers(self):
        g_config = super().configure_optimizers()
        d_opt = hydra.utils.instantiate(
            self.hparams.d_optimizer,
            self.discriminator.parameters(),
            _recursive_=False
        )
        d_sched: Optional[Any] = hydra.utils.instantiate(
            self.hparams.d_scheduler,
            optimizer=d_opt,
            _recursive_=False
        )
        if d_sched is None:
            return g_config, d_opt

        d_config = {
            "optimizer": d_opt,
            "lr_scheduler": {
                "scheduler": d_sched,
                "interval": "step",
                "frequency": self.hparams.d_step_frequency,
            },
        }
        return g_config, d_config