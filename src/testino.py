import deepspeed
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from vsr.models.VRT.modules.tmsa import RTMSA

class Debug(nn.Module):
    def __init__(self, phrase):
        super().__init__()
        self.phrase = phrase

    def forward(self, x):
        print(self.phrase, x.shape)
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

deepspeed.init_distributed(dist_backend="nccl", rank=0, world_size=1, distributed_port=50523)

MoE = deepspeed.moe.layer.MoE(
    hidden_size=img_size[1],
    expert=nn.Sequential(*
                         [
                             Debug("ENTERING MOE SHAPE:"),
                             Rearrange('n 1 (c d h) w -> n d h w c', d=window_size[0],
                                       c=embed_dims[len(scales) - 1] // top_k),
                             Debug("AFTER REARRANGE SHAPE:"),
                             nn.LayerNorm(embed_dims[len(scales) - 1] // top_k),
                             nn.Linear(embed_dims[len(scales) - 1] // top_k, embed_dims[len(scales)] // top_k),
                             Debug("AFTER LINEAR SHAPE:"),
                             Rearrange('n d h w c -> n c d h w'),
                             Debug("TRANSFORMER INPUT SHAPE:"),
                         ] +
                         [
                             RTMSA(dim=embed_dims[i] // top_k,
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
    num_experts=4,
    ep_size=num_gpus,
    k=2
).cuda()

x = torch.rand(1, 32, 6, 64, 64).cuda()

print("INPUT SHAPE:", x.shape)
out, _, _ = MoE(x)
print("FINAL SHAPE:", out.shape)
