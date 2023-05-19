import warnings
from concurrent.futures import ThreadPoolExecutor

import omegaconf
import pandas as pd

from core import PROJECT_ROOT
from core.utils import *

warnings.filterwarnings('ignore')
video_pd = []

def run(config):
    for fps in [15, 12, 10, 8, 6]:
        for crf in [30, 32, 34, 36, 38, 40]:
            print('Configuration: fps -> {} - crf -> {} '.format(fps, crf))
            pool = ThreadPoolExecutor(1)
            video_folder = os.path.join(config.lr_dir, f"fps={fps}_crf={crf}", "frames")
            output_folder = os.path.join(config.out_dir, os.path.basename(config.cfg_dir))
            video_paths = list(Path(video_folder).glob('*'))

            bpp = 0
            for video_lr_path in video_paths:
                video_name = os.path.basename(video_lr_path)
                video_hr_path = os.path.join(config.hr_dir, f"fps={fps}_crf=5", "frames", video_name)
                save_folder = os.path.join(output_folder, f"fps={fps}_crf={crf}", video_name)
                # Path(save_folder).mkdir(exist_ok=True, parents=True)

                video_hr = get_video(video_hr_path, pool)
                print("loaded")

                _, n_frames, c, h, w = video_hr.shape
                size_bits = (Path(config.lr_dir) / f"fps={fps}_crf={crf}" / "video" / video_name).stat().st_size * 8
                bpp += size_bits / (c * h * w * n_frames)

            video_pd.append({"bpp": bpp / len(video_paths), "fps": fps, "crf": crf})

    df = pd.read_csv(os.path.join(output_folder, f'{os.path.basename(config.cfg_dir)}.csv'))
    # df.merge(pd.DataFrame(video_pd)).to_csv(os.path.join(output_folder, f'{os.path.basename(config.cfg_dir)}.csv'))
    print(df.merge(pd.DataFrame(video_pd)))

    return output_folder

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(config: omegaconf.DictConfig):
    run(config)

if __name__ == "__main__":
    main()
