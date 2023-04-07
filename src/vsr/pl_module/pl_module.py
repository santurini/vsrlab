import logging
from typing import Any, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from core import PROJECT_ROOT
from core.losses import epe_loss
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import DictConfig
from torchvision.utils import make_grid

pylogger = logging.getLogger(__name__)

class LitVSR(pl.LightningModule):
    def __init__(
            self,
            model: DictConfig,
            loss: DictConfig,
            metric: DictConfig,
            *args,
            **kwargs
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        metric = hydra.utils.instantiate(metric, _recursive_=True, _convert_="partial")
        self.train_metric = metric.clone(postfix='/train')
        self.val_metric = metric.clone(postfix='/val')

        self.model = hydra.utils.instantiate(model, _recursive_=False)
        self.loss = hydra.utils.instantiate(loss, _recursive_=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, lr, hr):
        b, t, c, h, w = lr.shape
        sr, lq, _, _ = self(lr)

        loss = self.loss(sr, hr) + self.loss(lq, resize(hr, (h, w)))

        return {
            "lr": lr.detach(),
            "sr": sr.detach(),
            "hr": hr.detach(),
            "lq": lq.detach(),
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log_losses(
            step_out,
            "train"
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr, 'b t c h w -> (b t) c h w')
            )
        )

        return step_out

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log_losses(
            step_out,
            "val"
        )

        self.log_dict(
            self.val_metric(
                rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr, 'b t c h w -> (b t) c h w')
            )
        )

        if self.get_log_flag(batch_idx, self.hparams.log_interval):
            self.log_images(step_out)

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

    def log_losses(self, out, stage):
        out_dict = {}
        for key in out.keys():
            if 'loss' in key:
                new_key = '/'.join([key, stage])
                out_dict[new_key] = out[key].cpu().detach()

        self.log_dict(
            out_dict,
            prog_bar=True
        )

    def log_images(self, out):
        b, t, c, h, w = out["sr"].shape
        lr = resize(out["lr"][0][-1], (h, w))
        lq = resize(out["lq"][0][-1], (h, w))
        hr = out["hr"][0][-1]
        sr = out["sr"][0][-1].clamp(0, 1)

        grid = make_grid([lr, hr, lq, sr], nrow=2, ncol=2)
        self.logger.log_image(key='Input Images', images=[grid], caption=[f'Model Output: step {self.global_step}'])

    @staticmethod
    def get_log_flag(batch_idx, log_interval):
        flag = batch_idx % log_interval == 0
        return flag

class LitFlowVSR(LitVSR):
    def __init__(
            self,
            flow_loss: nn.Module,
            distillation = False,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.flow_loss = hydra.utils.instantiate(flow_loss, _recursive_=False)

        if distillation:
            self.distillation = distillation
            self.spynet = Spynet().requires_grad_(False)

    def step(self, lr, hr):
        b, t, c, h, w = lr.shape
        sr, lq, flow_f, flow_b = self(lr)

        pixel_loss = self.loss(sr, hr) + self.loss(lq, resize(hr, (h, w)))
        flow_loss = self.flow_loss(sr, hr)
        loss = pixel_loss + flow_loss

        if self.distillation:
            distillation_loss = self.flow_distillation(flow_f, hr) + self.flow_distillation(flow_b, hr, reverse=True)
            loss += distillation_loss

        return {
            "lr": lr.detach(),
            "sr": sr.detach(),
            "hr": hr.detach(),
            "lq": lq.detach(),
            "loss": loss,
            "pixel_loss": pixel_loss,
            "distillation_loss": distillation_loss,
            "flow_loss": flow_loss
        }

    def flow_distillation(self, flow, hr, reverse=False):
        img1 = hr[:, :-1, :, :, :].reshape(-1, c, hl, wl)
        img2 = hr[:, 1:, :, :, :].reshape(-1, c, hl, wl)

        if reverse:
            flow_hr = self.spynet(img1, img2)[-1]
        else:
            flow_hr = self.spynet(img2, img1)[-1]

        loss = 0
        for i in range(len(flow)):
            b, c, h, w = flow[i].shape
            loss += epe_loss(
                flow[i],
                resize(flow_hr, (h, w))
            )

        return loss

class LitGanVSR(LitVSR):
    def __init__(self,
                 discriminator: DictConfig,
                 adversarial_loss: DictConfig,
                 perceptual_loss: DictConfig,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.discriminator = hydra.utils.instantiate(discriminator, _recursive_=False)

        self.perceptual_loss: Optional[Any] = hydra.utils.instantiate(perceptual_loss, _recursive_=False)
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

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.cfg.nn.module,
        _recursive_=False,
    )

if __name__ == "__main__":
    main()
