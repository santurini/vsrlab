import torch.nn as nn
import torch.nn.functional as F
from core.modules.dct_transforms import EncoderDCT, DecoderIDCT
from core.modules.mlp import MlpMixer
from einops import rearrange

class MixerVSR(nn.Module):
    def __init__(self, img_size, mid_ch, it_blocks, patch_size,
                 exp, mix_blocks, conv_blocks, upscale):
        super().__init__()
        self.upscale = upscale
        height, width = img_size
        channels_dim = 3 * patch_size ** 2
        patches_dim = (height // patch_size) * (width // patch_size)
        self.encoder = EncoderDCT(patch_size)  # b (h/p)*(w/p) (c*p*p)
        self.mixer = MlpMixer(patches_dim, channels_dim, time_dim, exp, mix_blocks)  # b (h/p)*(w/p) (c*p*p)
        self.decoder = DecoderIDCT(mid_ch, patch_size, height, width)  # b c h w

    def forward(self, x1, x2):
        pass

