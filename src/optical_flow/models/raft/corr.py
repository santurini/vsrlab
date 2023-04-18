import torch
import torch.nn.functional as F
from optical_flow.models.raft.utils import bilinear_sampler

def correlation(coords, fmap1, fmap2, num_levels=4, radius=4):
    corr_pyramid = []
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())

    batch, h1, w1, dim, h2, w2 = corr.shape
    corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    corr_pyramid.append(corr)
    for i in range(num_levels - 1):
        corr = F.avg_pool2d(corr, 2, stride=2)
        corr_pyramid.append(corr)

    coords = coords.permute(0, 2, 3, 1)
    batch, h1, w1, _ = coords.shape

    out_pyramid = []
    for i in range(num_levels):
        corr = corr_pyramid[i]
        dx = torch.linspace(-radius, radius, 2 * radius + 1, device=coords.device)
        dy = torch.linspace(-radius, radius, 2 * radius + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
        delta_lvl = delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.view(batch, h1, w1, -1)
        out_pyramid.append(corr)

    out = torch.cat(out_pyramid, dim=-1)
    return out.permute(0, 3, 1, 2).contiguous().float()