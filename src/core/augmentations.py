import io
from random import randint, choice

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
    def __init__(self, codec, crf, fps, bitrate=None):
        super().__init__()
        self.codec = choice(codec) if len(codec) > 1 else codec
        self.crf = str(randint(crf[0], crf[1])) if len(crf) == 2 else str(crf)
        self.fps = str(randint(fps[0], fps[1])) if len(fps) == 2 else str(fps)

    def forward(self, video):
        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.height = video[0].shape[-2]
            stream.width = video[0].shape[-1]
            stream.pix_fmt = 'yuv420p'
            # stream.bit_rate = self.bitrate
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

class Mirroring(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x):
        flipped = x.flip(0)
        extended = torch.concat((x, flipped), 0)
        return extended