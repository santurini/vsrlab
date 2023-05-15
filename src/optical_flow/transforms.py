import random
from typing import Union, Tuple

import torch
import torchvision.transforms.functional as F
from kornia.enhance import normalize

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sequence, optical_flow):
        for t in self.transforms:
            sequence, optical_flow = t(sequence, optical_flow)
        return sequence, optical_flow

class Resize(object):
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self,
                 frames,
                 optical_flow):
        frames = F.resize(frames, (self.height, self.width))
        optical_flow = F.resize(optical_flow, (2, self.height, self.width))

        return frames, optical_flow

class RandomRotation(object):
    def __init__(self, minmax: Union[Tuple[int, int], int]) -> None:
        self.minmax = minmax
        if isinstance(minmax, int):
            self.minmax = (-minmax, minmax)

    def __call__(self,
                 frames,
                 optical_flow):
        p = random.uniform(0, 1) > 1 - self.p
        if p:
            angle = random.randint(*self.minmax)
            frames = F.rotate(frames, angle)
            optical_flow = F.rotate(optical_flow, angle)

        return frames, optical_flow

class RandomHorizontalFlip(object):
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self,
                 frames,
                 optical_flow):
        p = random.uniform(0, 1) > 1 - self.p
        if p:
            frames = F.hflip(frames)
            optical_flow = F.hflip(optical_flow, angle)

        return frames, optical_flow

class RandomVerticalFlip(object):
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self,
                 frames,
                 optical_flow):
        p = random.uniform(0, 1) > 1 - self.p
        if p:
            frames = F.vflip(frames)
            optical_flow = F.vflip(optical_flow, angle)

        return frames, optical_flow

class Normalize(object):
    def __init__(self,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self,
                 frames: Tuple[torch.Tensor, torch.Tensor],
                 optical_flow: torch.Tensor):
        frames = normalize(frames, self.mean, self.std)
        return frames, optical_flow

class RandomVideoCompression(object):
    def __init__(self, codec, crf, fps):
        self.codec = choice(codec) if len(codec) > 1 else codec[0]
        self.crf = str(randint(crf[0], crf[1])) if len(crf) == 2 else str(crf[0])
        self.fps = str(randint(fps[0], fps[1])) if len(fps) == 2 else str(fps[0])

    def __call__(self, frames, optical_flow):
        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.height = frames[0].shape[-2]
            stream.width = frames[0].shape[-1]
            stream.pix_fmt = 'yuv420p'
            stream.options = {"crf": self.crf}
            for frame in frames:
                frame = av.VideoFrame.from_image(F.to_pil_image(frame))
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(F.to_tensor(frame.to_image()))

        return torch.stack(outputs), optical_flow
