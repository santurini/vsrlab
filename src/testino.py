import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from fmoe.layers import FMoE

from vsr.models.VRT.modules.tmsa import RTMSA

class Debug(nn.Module):
    def __init__(self, phrase):
        super().__init__()
        self.phrase = phrase

    def forward(self, x):
        print(x)
        return x

num_gpus = 1
top_k = 2
num_experts = 4
scales = [1, 2, 4, 2, 1]
embed_dims = [32, 32, 32, 32, 32, 32, 32]
depths = [4, 4, 4, 4, 4, 4, 4]
img_size = [6, 64, 64]
window_size = [6, 8, 8]
indep_reconsts = [-2, -1]
num_heads = [4, 4, 4, 4, 4, 4, 4]
mul_attn_ratio = 0.75
mlp_ratio = 2.
qkv_bias = True
qk_scale = None
drop_path_rate = 0.2
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
norm_layer = nn.LayerNorm

MoE = FMoE(
    num_expert=4,
    d_model=embed_dims[len(scales) - 1],
    world_size=1,
    top_k=2,
    expert=nn.Sequential(*
                         [
                             Debug("MoE Input"),
                             Rearrange('n c d h w-> n d h w c'),
                             Debug("After Rearrange"),
                             nn.LayerNorm(embed_dims[len(scales) - 1]),
                             nn.Linear(embed_dims[len(scales) - 1], embed_dims[len(scales)]),
                             Debug("After Linear"),
                             Rearrange('n d h w c -> n c d h w'),
                         ] +
                         [
                             RTMSA(dim=embed_dims[i],
                                   input_resolution=img_size,
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=[1, window_size[1],
                                                window_size[2]] if i in indep_reconsts else window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   norm_layer=norm_layer
                                   ) for i in range(len(scales), len(depths))
                         ]
                         )
).cuda()

x = torch.rand(1, 6, 64, 64, 32).cuda()

print("INPUT SHAPE:", x.shape)
out, _, _ = MoE(x)
print("FINAL SHAPE:", out.shape)
