import logging
from typing import Any, Sequence, Tuple, Union, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from core import PROJECT_ROOT
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import DictConfig

pylogger = logging.getLogger(__name__)

class LitSR(pl.LightningModule):
    def __init__(self,
                 model: DictConfig,
                 loss: DictConfig,
                 *args,
                 **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        metric = hydra.utils.instantiate(self.hparams.metric, _recursive_=True)
        self.train_metric = metric.clone()
        self.val_metric = metric.clone()

        self.model = hydra.utils.instantiate(model, _recursive_=False)

        self.loss = hydra.utils.instantiate(loss, _recursive_=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, lr, hr):
        sr = self(lr)
        loss = self.loss(sr, hr)
        return {"sr": sr.detach(), "lq": lr, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log(
            "loss/train",
            step_out["loss"].cpu().detach(),
            prog_bar=True,
        )

        metric_dict = self.train_metric(
            step_out["sr"].clamp(0, 1),
            hr
        )
        self.log_dict(
            metric_dict,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int):
        lr, hr = batch
        step_out = self.step(lr, hr)
        self.log(
            "loss/val",
            step_out["loss"].cpu().detach(),
            prog_bar=True,
        )

        metric_dict = self.val_metric(
            step_out["sr"].clamp(0, 1),
            hr
        )
        self.log_dict(
            metric_dict,
        )

        if self.get_log_flag(batch_idx, self.hparams.log_interval):
            self.log_images(step_out["lq"], step_out["sr"], hr)

        return step_out

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            self.parameters(),
            _recursive_=False
        )
        scheduler: Optional[Any] = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer,
            _recursive_=False,
        )
        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1},
             }

    def log_images(self, lr, sr, hr):
        t_log = lr.shape[0] if lr.shape[0] < 5 else self.hparams.log_k_images
        lr = lr[:t_log].detach().cpu()
        hr = hr[:t_log].detach().cpu()
        sr = sr[:t_log].clamp(0, 1).detach().cpu()
        psnr = ['PSNR: ' + str(self.val_metric(i, j)['PeakSignalNoiseRatio'].detach().cpu().numpy().round(2)) for i, j in zip(sr, hr)]
        self.logger.log_image(key='Imput Image', images=[i for i in lr], caption=[f'inp_img_{i + 1}' for i in range(t_log)])
        self.logger.log_image(key='Ground Truths', images=[i for i in hr], caption=[f'gt_img_{i+1}' for i in range(t_log)])
        self.logger.log_image(key='Predicted Images', images=[i for i in sr], caption=psnr)

    @staticmethod
    def get_log_flag(batch_idx, log_interval):
        flag = batch_idx%log_interval==0
        return flag


class LitVSR(LitSR):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log(
            "loss/train",
            step_out["loss"].cpu().detach(),
            prog_bar=True,
        )

        metric_dict = self.train_metric(
            rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
            rearrange(hr, 'b t c h w -> (b t) c h w')
        )
        self.log_dict(
            metric_dict,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log_dict(
            "loss/val",
            step_out["loss"].cpu().detach(),
            prog_bar=True,
        )

        metric_dict = self.val_metric(
            rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
            rearrange(hr, 'b t c h w -> (b t) c h w')
        )
        self.log_dict(
            metric_dict,
        )

        print(metric_dict)

        if self.get_log_flag(batch_idx, self.hparams.log_interval):
            self.log_images(step_out["lq"], step_out["sr"], hr)

        return step_out

    def log_images(self, lr, sr, hr):
        t_log = 5 if not self.hparams.log_k_images else self.hparams.log_k_images
        lr = lr[0][:t_log]
        hr = hr[0][:t_log]
        sr = sr[0][:t_log].clamp(0, 1)
        psnr = ['PSNR: ' + str(self.val_metric(i, j)['PeakSignalNoiseRatio'].detach().cpu().numpy().round(2)) for i, j in zip(sr, hr)]
        self.logger.log_image(key='Input Images', images=[i for i in lr], caption=[f'inp_frame_{i + 1}' for i in range(t_log)])
        self.logger.log_image(key='Ground Truths', images=[i for i in hr], caption=[f'gt_frame_{i+1}' for i in range(t_log)])
        self.logger.log_image(key='Predicted Images', images=[i for i in sr], caption=psnr)


class LitRealVSR(LitVSR):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def step(self, lr, hr):
        b, t, c, h, w = lr.shape
        sr, lq = self(lr)
        loss = self.loss(sr, hr) + self.loss(lq, resize(hr, (h, w)))
        return {"sr": sr.detach(), "lq": lq.detach(), "loss": loss}

class LitRealGanVSR(LitRealVSR):
    def __init__(self,
                 discriminator: DictConfig,
                 adversarial_loss: DictConfig,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.discriminator = hydra.utils.instantiate(discriminator, _recursive_=False)

        self.perceptual_loss: Optional[Any] = hydra.utils.instantiate(self.hparams.perceptual_loss, _recursive_=False)
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
             "loss/train/generator_fake": disc_fake_loss.cpu().detach(),},
        )
        self.train_metric(
            rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
            rearrange(hr, 'b t c h w -> (b t) c h w')
        )
        self.log_dict(
            self.train_metric,
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

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        _recursive_=False,
    )

if __name__ == "__main__":
    main()