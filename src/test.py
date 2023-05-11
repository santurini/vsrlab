import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import omegaconf
import pandas as pd
from torchvision.utils import save_image

from core import PROJECT_ROOT
from core.utils import *

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

@torch.no_grad()
def run(config):
    rank, local_rank, world_size = (0, 0, 1)
    device = torch.device("cuda:{}".format(local_rank))

    cfg = OmegaConf.load(os.path.join(config.cfg_dir, "config.yaml"))
    ckpt_path = os.path.join(config.cfg_dir, "last.ckpt")

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model = build_model(cfg.train.model, device, local_rank, False, ckpt_path)

    print('build metrics and losses ...')
    metric, video_pd = build_metric(config.metric).to(device), []

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Testing ...".format(rank, local_rank))
    pool = ThreadPoolExecutor(config.num_workers)

    for fps in [15, 12, 10, 8, 6]:
        for crf in [30, 32, 34, 36, 38, 40]:
            print('Configuration: fps -> {} - crf -> {} '.format(fps, crf))
            video_folder = os.path.join(config.lr_dir, f"fps={fps}_crf={crf}", "frames")
            output_folder = os.path.join(config.out_dir, os.path.basename(config.cfg_dir))
            video_paths = list(Path(video_folder).glob('*'))
            video_metrics = {k: 0 for k in config.metric.metrics}

            for video_lr_path in video_paths:
                model.eval();
                dt = time.time()

                video_name = os.path.basename(video_lr_path)
                video_hr_path = os.path.join(config.hr_dir, f"fps={fps}_crf=5", "frames", video_name)
                save_folder = os.path.join(output_folder, f"fps={fps}_crf={crf}", video_name)
                Path(save_folder).mkdir(exist_ok=True, parents=True)

                video_hr, video_lr = get_video(video_hr_path, pool).to(device), get_video(video_lr_path, pool).to(
                    device)

                outputs = []
                for i in range(0, video_lr.size(1), config.window_size):
                    lr, hr = video_lr[:, i:i + config.window_size, ...].to(device), \
                        video_hr[:, i:i + config.window_size, ...].to(device)

                    sr, _ = model(lr)
                    outputs.append(sr)

                dt = time.time() - dt
                print(f"Inference Time --> {dt:2f}")

                outputs = torch.cat(outputs, dim=1)

                pool.map(
                        lambda x: save_image(x[1], os.path.join(save_folder, "img{:05d}.png".format(x[0]))),
                        enumerate(outputs[0]), 
                )

                video_metrics = running_metrics(video_metrics, metric, outputs, video_hr)

            video_pd.append({"fps": fps, "crf": crf} | {k: v / len(video_paths) for k, v in video_metrics.items()})

    pd.DataFrame(video_pd).to_csv(os.path.join(output_folder, 'metrics.csv'))

    return output_folder

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
