import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import omegaconf
import pandas as pd
import torch
from torchvision.utils import save_image

from core.utils import (
    build_test_model,
    build_metric,
    get_video,
    running_metrics
)

warnings.filterwarnings('ignore')

C, H, W = 3, 480, 640

def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def init_model(config, checkpoint=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    return model

@torch.no_grad()
def run(config: omegaconf.DictConfig):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    if os.path.basename(config.cfg_dir) == "basic_og":
        cfg = os.path.join(config.cfg_dir, "realbasicvsr_x4.py")
        ckpt_path = os.path.join(config.cfg_dir, "RealBasicVSR_x4.pth")
        model = init_model(cfg, ckpt_path).to(device)
    else:
        cfg = omegaconf.OmegaConf.load(os.path.join(config.cfg_dir, "config.yaml"))
        ckpt_path = os.path.join(config.cfg_dir, "last.ckpt")
        model = build_test_model(cfg.train.model, device, ckpt_path)

    print('build metrics and losses ...')
    metric, video_pd = build_metric(config.metric).to(device), []

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    pool = ThreadPoolExecutor(config.num_workers)

    for fps in [6, 8, 10]:
        for crf in [30, 32, 34]:
            print('Configuration: fps --> {} - crf -> {}\n'.format(fps, crf))
            video_folder = os.path.join(config.lr_dir, f"fps={fps}_crf={crf}", "frames")
            output_folder = os.path.join(config.out_dir, os.path.basename(config.cfg_dir))
            video_paths = list(Path(video_folder).glob('*'))
            metrics, bpp, cf = {k: 0 for k in config.metric.metrics}, 0, 0

            for video_lr_path in video_paths:
                model.eval()
                dt = time.time()

                video_name = os.path.basename(video_lr_path)
                video_hr_path = os.path.join(config.hr_dir, f"fps={fps}_crf=5", "frames", video_name)
                save_folder = os.path.join(output_folder, f"fps={fps}_crf={crf}", video_name)
                Path(save_folder).mkdir(exist_ok=True, parents=True)

                print('Loading Video --> {}'.format(video_name))
                video_hr, video_lr = get_video(video_hr_path, pool), get_video(video_lr_path, pool)

                F = video_hr.size(1)
                size_bits_orig = (Path(config.hr_dir) / f"fps={fps}_crf=5" / "video" / video_name).stat().st_size * 8
                size_bits_comp = (Path(
                    config.lr_dir) / f"fps={fps}_crf={crf}" / "video" / video_name).stat().st_size * 8

                cf += size_bits_comp / size_bits_orig
                bpp += size_bits_comp / (C * H * W * F)

                outputs = []
                windows = list(range(0, video_lr.size(1), config.window_size))
                video_metrics, norm_factor = {k: 0 for k in config.metric.metrics}, len(windows)
                for i in windows:
                    lr, hr = video_lr[:, i:i + config.window_size, ...].to(device, non_blocking=True), \
                        video_hr[:, i:i + config.window_size, ...].to(device, non_blocking=True)

                    sr, _ = model(lr)
                    outputs.append(sr)

                    video_metrics = running_metrics(video_metrics, metric, sr, hr)

                outputs = torch.cat(outputs, dim=1)

                print('Saving Video to --> {}'.format(save_folder))
                list(pool.map(
                    lambda x: save_image(x[1], os.path.join(save_folder, "img{:05d}.png".format(x[0]))),
                    enumerate(outputs[0]),
                ))

                video_metrics = {k: v / norm_factor for k, v in video_metrics.items()}
                metrics = {k: (metrics[k] + video_metrics[k]) for k in set(metrics) & set(video_metrics)}

                dt = time.time() - dt
                print(f"Inference Time --> {dt:2f}\n")

            video_pd.append(
                {"cf": cf / len(video_paths), "bpp": bpp / len(video_paths), "fps": fps, "crf": crf} |
                {k: v / len(video_paths) for k, v in metrics.items()}
            )

    results = pd.DataFrame(video_pd)
    results.to_csv(os.path.join(output_folder, f'{os.path.basename(config.cfg_dir)}.csv'), index=False)

    return results

def test_video(lq, model, cfg, config):
    '''test the video as a whole or as clips (divided temporally). '''

    num_frame_testing = config.tile[0]
    if num_frame_testing:
        # test as multiple clips if out-of-memory
        sf = 4
        num_frame_overlapping = config.tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx:d_idx + num_frame_testing, ...]
            out_clip = test_clip(lq_clip, model, cfg, config)
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, :num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, :num_frame_overlapping // 2, ...] *= 0

            E[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip)
            W[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)

    return output

def test_clip(lq, model, cfg, config):
    ''' test the clip as a whole or as patches. '''

    sf = 4
    window_size = cfg.train.model.window_size
    size_patch_testing = config.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = config.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size // 2:, :] *= 0
                        out_patch_mask[..., -overlap_size // 2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size // 2:] *= 0
                        out_patch_mask[..., :, -overlap_size // 2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size // 2, :] *= 0
                        out_patch_mask[..., :overlap_size // 2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size // 2] *= 0
                        out_patch_mask[..., :, :overlap_size // 2] *= 0

                E[..., h_idx * sf:(h_idx + size_patch_testing) * sf, w_idx * sf:(w_idx + size_patch_testing) * sf].add_(
                    out_patch)
                W[..., h_idx * sf:(h_idx + size_patch_testing) * sf, w_idx * sf:(w_idx + size_patch_testing) * sf].add_(
                    out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old * sf, :w_old * sf]

    return output

if __name__ == '__main__':
    main()
