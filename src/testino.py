import torch
import torch.nn as nn
from fmoe.transformer import FMoETransformerMLP

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

MoE = FMoETransformerMLP(
    num_expert=4,
    d_model=embed_dims[len(scales) - 1],
    d_hidden=embed_dims[len(scales) - 1]
).cuda()

x = torch.rand(1, 6, 64, 64, 32).cuda()

print("INPUT SHAPE:", x.shape)
out = MoE(x)
print("FINAL SHAPE:", out.shape)
