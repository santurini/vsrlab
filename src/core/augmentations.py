import io
import os
from random import randint, choice
from pathlib import Path

import av
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

class RandomJPEGCompression(nn.Module):
    def __init__(self, quality):
        super().__init__()
        self.q = randint(quality[0], quality[1]) if len(quality) == 2 else quality

    def forward(self, in_batch):
        if in_batch.dim() == 3:
            return self.torch_add_compression(in_batch.detach().clamp(0, 1), self.q)
        return self.torch_batch_add_compression(in_batch.detach().clamp(0, 1), self.q).type_as(in_batch)

    @staticmethod
    def pil_add_compression(pil_img: Image.Image, q: int) -> Image.Image:
        # BytesIO: just like an opened file, but in memory
        with io.BytesIO() as buffer:
            # do the actual compression
            pil_img.save(buffer, format='JPEG', quality=q)
            buffer.seek(0)
            with Image.open(buffer) as compressed_img:
                compressed_img.load()  # keep image in memory after exiting the `with` block
                return compressed_img

    def torch_add_compression(self, in_tensor: torch.Tensor, q: int) -> torch.Tensor:
        pil_img = F.to_pil_image(in_tensor)
        compressed_img = self.pil_add_compression(pil_img, q=q)
        return F.to_tensor(compressed_img).type_as(in_tensor)

    def torch_batch_add_compression(self, in_batch: torch.Tensor, q: int) -> torch.Tensor:
        return torch.stack([self.torch_add_compression(elem, q) for elem in in_batch])

class RandomVideoCompression(nn.Module):
    def __init__(self, codec, crf, fps):
        super().__init__()
        self.codec = choice(codec) if len(codec) > 1 else codec
        self.crf = str(randint(crf[0], crf[1])) if len(crf) == 2 else str(crf[0])
        self.fps = str(randint(fps[0], fps[1])) if len(fps) == 2 else str(fps[0])

    def forward(self, video):
        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.height = video[0].shape[-2]
            stream.width = video[0].shape[-1]
            stream.pix_fmt = 'yuv420p'
            stream.options = {"crf": self.crf}
            for frame in video:
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

        return torch.stack(outputs)

def read_video(path):
    with av.open(path) as container:
        assert container.streams.video, f"not a video: {path}"
        container.streams.video[0].thread_type = "AUTO"
        frames = [frame for frame in container.decode(video=0)]
        rate = str(container.streams.video[0].average_rate.numerator)
        height = container.streams.video[0].height
        width = container.streams.video[0].width
        codec =  container.streams.video[0].codec.name

    return frames, codec, rate, height, width

def write_video(path, frames, codec, rate, crf, height, width):
    with av.open(path, 'w') as container:
        stream = container.add_stream(codec, rate=rate)
        stream.height = height
        stream.width = width
        stream.pix_fmt = 'yuv420p'
        stream.options = {"crf": str(crf)}
        for frame in frames:
            frame.pict_type = 'NONE'
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

def compress_video(path_hr, path_lr, crf, scale_factor):

    frames_hr, codec, rate, height, width =  read_video(path_hr)

    assert height%scale_factor==0, f"{height=} should be divisible by scale factor"
    assert width%scale_factor==0, f"{width=} should be divisible by scale factor"

    write_video(path_lr, frames_hr, codec, rate, crf, height//scale_factor, width//scale_factor)

def compress_video_folder(folder, crf, scale_factor):
    os.mkdir(os.path.join(folder, f'lr_crf_{crf}'))
    paths = Path(folder).glob('hr/*')
    for video in paths:
        file_name = f'lr_crf_{crf}/{video.name}'
        compress_video(str(video), str(Path(folder) / Path(file_name)), crf, scale_factor)

class Mirroring(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x):
        flipped = x.flip(0)
        extended = torch.concat((x, flipped), 0)
        return extended
