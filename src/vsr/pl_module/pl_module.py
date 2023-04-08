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

        self.loss = hydra.utils.instantiate(loss, _recursive_=True, _convert_="partial")

        self.model = hydra.utils.instantiate(model, _recursive_=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, lr, hr):
        sr, lq, _, _ = self(lr)

        args = {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "lq": lq
        }

        args = self.loss(args)
        print(args["L1Loss"])
        return args

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr)

        self.log_losses(
            step_out,
            "train",
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].detach().clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr.detach(), 'b t c h w -> (b t) c h w')
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
                rearrange(step_out["sr"].detach().clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr.detach(), 'b t c h w -> (b t) c h w')
            )
        )

        if self.get_log_flag(batch_idx, self.hparams.log_interval):
            self.log_images(step_out)

        return step_out

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            self.filter_params(self.hparams.group_lr),
            _recursive_=False,
            _convert_="partial"
        )

        scheduler: Optional[Any] = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer,
            _recursive_=False,
            _convert_="partial"
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

    def filter_params(
            self,
            group: DictConfig
    ):
        group_0 = list(map(lambda x: x[1], list(
            filter(lambda kv: group.group in kv[0], self.model.named_parameters()))))
        group_1 = list(map(lambda x: x[1], list(
            filter(lambda kv: not group.group in kv[0], self.model.named_parameters()))))

        params = [
            {"params": group_0, "lr": group.lr[0]},
            {"params": group_1, "lr": group.lr[1]}
        ]

        return params

    def log_losses(self, out, stage):
        out_dict = {}
        for key in out.keys():
            if 'loss' in key:
                new_key = '/'.join([key, stage])
                out_dict[new_key] = out[key].cpu().detach()

        self.log_dict(
            out_dict,
            on_epoch=True,
            prog_bar=False
        )

    def log_images(self, out):
        b, t, c, h, w = out["sr"].shape
        lr = resize(out["lr"][0][-1], (h, w)).detach()
        lq = resize(out["lq"][0][-1], (h, w)).detach()
        hr = out["hr"][0][-1].detach()
        sr = out["sr"][0][-1].detach().clamp(0, 1)

        grid = make_grid([lr, hr, lq, sr], nrow=2, ncol=2)
        self.logger.log_image(key='Input Images', images=[grid], caption=[f'Model Output: step {self.global_step}'])

    @staticmethod
    def get_log_flag(batch_idx, log_interval):
        flag = batch_idx % log_interval == 0
        return flag

class LitFlowVSR(LitVSR):
    def __init__(
            self,
            distillation = False,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        if distillation:
            self.distillation = distillation
            self.spynet = Spynet().requires_grad_(False)

    def step(self, lr, hr):
        sr, lq, flow_f, flow_b = self(lr)

        args = {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "lq": lq,
            "flow_f": flow_f,
            "flow_b": flow_b
        }

        args = self.loss(args)

        return args

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr)

        if self.distillation:
            distillation_loss = self.flow_distillation(step_out["flow_f"], step_out["hr"]) + \
                                self.flow_distillation(step_out["flow_b"], step_out["hr"], reverse=True)
            step_out["distillation_loss"] = distillation_loss
            step_out["loss"] += distillation_loss

        self.log_losses(
            step_out,
            "train",
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr, 'b t c h w -> (b t) c h w')
            )
        )

        return step_out

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
