import logging
from typing import Any, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from core import PROJECT_ROOT
from core.losses import rmse_loss, AdversarialLoss, PerceptualLoss
from einops import rearrange
from kornia.geometry.transform import resize
from omegaconf import DictConfig
from torchvision.utils import make_grid

pylogger = logging.getLogger(__name__)

class LitBase(pl.LightningModule):
    def __init__(
            self,
            model: DictConfig,
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

    def forward(
            self,
            lr: torch.Tensor,
            hr: torch.Tensor,
            test: bool
    ):
        return self.model(lr, hr, test)

    def step(self, lr, hr, test):
        sr, loss = self(lr, hr, test)

        args = {
            "lr": lr,
            "sr": sr,
            "hr": hr,
            "loss": loss
        }

        return args

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr, test=False)

        self.log_losses(
            step_out,
            "train",
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].detach().clamp(0, 1), 'b c t h w -> (b t) c h w').contiguous(),
                rearrange(hr.detach(), 'b c t h w -> (b t) c h w').contiguous()
            )
        )

        print(self.model.rrdbnet.weight[0][0][0])

        return step_out["loss"]

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        step_out = self.step(lr, hr, test=True)

        self.log_losses(
            step_out,
            "val"
        )

        self.log_dict(
            self.val_metric(
                rearrange(step_out["sr"].detach().clamp(0, 1), 'b c t h w -> (b t) c h w').contiguous(),
                rearrange(hr.detach(), 'b c t h w -> (b t) c h w').contiguous()
            )
        )

        if self.get_log_flag(batch_idx, self.hparams.log_interval):
            self.log_images(step_out)

        return step_out["loss"]

    def configure_optimizers(self):
        opt_config = self._configure_optimizers(
            self.model,
            self.hparams.optimizer,
            self.hparams.scheduler,
            self.hparams.set_lr,
        )
        return opt_config

    def _configure_optimizers(
            self,
            model: nn.Module,
            optim_cfg: DictConfig,
            sched_cfg: DictConfig = None,
            set_lr: Any = None,
    ):
        pylogger.info(f"Configuring optimizer for <{model.__class__.__name__}>")
        if not set_lr:
            pylogger.info("Nothing to filter here!")
            parameters = [{"params": model.parameters()}]
        else:
            parameters = self.filter_params(
                set_lr,
                model
            )

        optimizer = hydra.utils.instantiate(
            optim_cfg,
            parameters,
            _recursive_=False,
            _convert_="partial"
        )

        if sched_cfg is None:
            return optimizer

        scheduler: Optional[Any] = hydra.utils.instantiate(
            sched_cfg,
            optimizer,
            _recursive_=False,
            _convert_="partial"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1},
        }

    @staticmethod
    def filter_params(
            set_lr: DictConfig,
            model: nn.Module
    ):
        assert (
            len(set_lr.lrs) == len(set_lr.groups),
            "For {} groups are expected the same number of learning rates but found {}".format(
                len(set_lr.groups), len(set_lr.lrs)
            )
        )

        parameters = []
        for group, lr in zip(set_lr.groups, set_lr.lrs):
            pylogger.info(f"Setting learning rate for parameters in <{group}> to <{lr}>")
            params = list(map(lambda x: x[1], list(
                filter(lambda kv: group in kv[0], model.named_parameters()))))
            parameters.append({"params": params, "lr": lr})

        params = list(map(lambda x: x[1], list(
            filter(lambda kv: not any(g in kv[0] for g in set_lr.groups), model.named_parameters()))))
        parameters.append({"params": params})

        return parameters

    def log_losses(self, out, stage):
        out_dict = {}
        for key in out.keys():
            if 'loss' in key.lower():
                new_key = '/'.join([key, stage])
                out_dict[new_key] = out[key]

        self.log_dict(
            out_dict,
            on_epoch=True,
            prog_bar=False
        )

    def log_images(self, out):
        b, t, c, h, w = out["sr"].shape
        lr = resize(out["lr"][0, :, -1, :, :], (h, w)).detach()
        hr = out["hr"][0, :, -1, :, :].detach()
        sr = out["sr"][0, :, -1, :, :].detach().clamp(0, 1)

        grid = make_grid([lr, sr, hr], nrow=3, ncol=1)
        self.logger.log_image(key='Input Images', images=[grid], caption=[f'Model Output: step {self.global_step}'])

    @staticmethod
    def get_log_flag(batch_idx, log_interval):
        flag = batch_idx % log_interval == 0
        return flag

class LitGan(LitBase):
    def __init__(self,
                 discriminator: DictConfig,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.automatic_optimization = False

        self.discriminator = hydra.utils.instantiate(discriminator, _recursive_=False)

        self.perceptual: nn.Module = PerceptualLoss()
        self.adversarial: nn.Module = AdversarialLoss()

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        self.toggle_optimizer(opt_g)
        step_out = self.generator_step(batch)
        self._optim_step(opt_g, step_out["loss"])
        self.toggle_optimizer(opt_d)
        loss = self.discriminator_step((step_out["sr"], step_out["hr"]))
        self._optim_step(opt_d, loss)

    def generator_step(self, batch):
        lr, hr = batch
        b, t, c, h, w = hr.shape
        step_out = self.step(lr, hr)
        perceptual_loss = self.perceptual(step_out["sr"], hr)
        disc_sr = self.discriminator(step_out["sr"].view(-1, c, h, w))
        disc_fake_loss = self.adversarial(disc_sr, 1, False)
        loss = step_out['loss'] + perceptual_loss + disc_fake_loss

        self.log_dict(
            {"loss/train/generator": loss.cpu().detach(),
             "loss/train/generator_pixel": step_out['loss'].cpu().detach(),
             "loss/train/generator_perceptual": perceptual_loss.cpu().detach(),
             "loss/train/generator_adversarial": disc_fake_loss.cpu().detach(), },
        )

        self.log_dict(
            self.train_metric(
                rearrange(step_out["sr"].clamp(0, 1), 'b t c h w -> (b t) c h w'),
                rearrange(hr, 'b t c h w -> (b t) c h w')
            ),
        )

        return loss

    def discriminator_step(self, batch):
        sr, hr = batch
        sr = rearrange(sr, 'b t c h w -> (b t) c h w')
        hr = rearrange(hr, 'b t c h w -> (b t) c h w')
        disc_hr = self.discriminator(hr)
        disc_true_loss = self.adversarial(disc_hr, 1, True)
        disc_sr = self.discriminator(sr.detach())
        disc_fake_loss = self.adversarial(disc_sr, 0, True)
        loss = disc_fake_loss + disc_true_loss

        self.log_dict(
            {"loss/train/discriminator": loss.cpu().detach(),
             "loss/train/discriminator_adversarial_fake": disc_fake_loss.cpu().detach(),
             "loss/train/discriminator_adversarial_true": disc_true_loss.cpu().detach()},
        )

        return loss

    def _optim_step(self, optimizer, loss):
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)

    def configure_optimizers(self):
        g_config = self._configure_optimizers(
            self.model,
            self.hparams.optimizer.generator,
            self.hparams.scheduler.generator,
            self.hparams.set_lr,
        )

        d_config = self._configure_optimizers(
            self.discriminator,
            self.hparams.optimizer.discriminator,
            self.hparams.scheduler.discriminator,
            self.hparams.d_set_lr,
        )

        return g_config, d_config

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.cfg.nn.module,
        _recursive_=False,
    )

if __name__ == "__main__":
    main()
