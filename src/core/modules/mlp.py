import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, dim, in_dim):
        super(Mlp, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, in_dim),
                                 nn.GELU(),
                                 nn.Linear(in_dim, dim))

    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, patches_dim, channels_dim, time_dim, exp):
        super(MixerBlock, self).__init__()
        self.time_mlp = Mlp(time_dim, exp * time_dim)
        self.patches_mlp = Mlp(patches_dim, exp * patches_dim)
        self.channels_mlp = Mlp(channels_dim, exp * channels_dim)

    def forward(self, x):
        x = (self.channels_mlp(x) + x).permute(0, 1, 3, 2)
        x = (self.patches_mlp(x) + x).permute(0, 2, 3, 1)
        x = (self.time_mlp(x) + x).permute(0, 3, 2, 1)
        return x

class MlpMixer(nn.Module):
    def __init__(self, patches_dim, channels_dim, time_dim, exp, blocks):
        super().__init__()
        self.mixer = nn.Sequential(*[MixerBlock(patches_dim, channels_dim, time_dim, exp) for _ in range(blocks)])

    def forward(self, x):
        return self.mixer(x)