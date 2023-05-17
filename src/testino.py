import deepspeed
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from vsr.models.VRT.modules.tmsa import RTMSA

class Debug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("IM HERE:", x.shape)
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

MoE = deepspeed.moe.layer.MoE(
    hidden_size=img_size[1],
    expert=nn.Sequential(*
                         [
                             Debug(),
                             Rearrange('n g (c e d h) w -> n d h (w g) (c e)', d=6, e=top_k,
                                       c=embed_dims[len(scales) - 1] // top_k),
                             Debug(),
                             nn.LayerNorm(embed_dims[len(scales) - 1]),
                             Debug(),
                             nn.Linear(embed_dims[len(scales) - 1], embed_dims[len(scales)]),
                             Debug(),
                             Rearrange('n d h (w g) (c e) -> n (c e) d h (w g)', e=top_k, g=num_gpus),
                             Debug(),
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
                         ),
    num_experts=num_experts,
    ep_size=num_gpus,
    k=top_k
)

x = torch.rand(1, 32, 6, 64, 64)

out, _, _ = stage6(x)
print(out.shape)
