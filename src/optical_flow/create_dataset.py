import warnings
from pathlib import Path

import ptlflow
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize

warnings.filterwarnings('ignore')

teacher = ptlflow.get_model('gmflow', pretrained_ckpt="kitti")
for p in teacher.parameters():
    p.requires_grad = False

teacher.cuda()

SAVE_DIR = "/home/aghinassi/Desktop/Flow"
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

with torch.no_grad():
    videos = list(Path('/home/aghinassi/Desktop/MergedVSR').glob('*'))
    for i, video in enumerate(videos):
        print("Video {} / {}".format(i, len(videos)))
        frames = list(sorted(video.glob('*')))
        couples = zip(frames[:-1], frames[1:])
        for c in couples:
            filename = '_'.join([c[0].parent.stem, c[0].stem, c[1].stem])
            save_path = f"{SAVE_DIR}/{filename}.pt"
            inputs = {
                "images": torch.stack(
                    [resize(to_tensor(Image.open(c[0])), size=(384, 512)),
                     resize(to_tensor(Image.open(c[1])), size=(384, 512))]
                ).unsqueeze(0).cuda()
            }
            flow = teacher(inputs)["flows"].squeeze(0).squeeze(0)
            torch.save(flow.detach(), save_path)
