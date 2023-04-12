import torch
import torch.nn.functional as F
from src.optical_flow.modules.raft.utils import bilinear_sampler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

def corr_torch(fmap1, fmap2, num_levels, radius):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())

    batch, h1, w1, dim, h2, w2 = corr.shape
    corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    corr_pyramid = [corr]
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

def alt_corr_cuda(fmap1, fmap2, num_levels, radius):
    pyramid = [(fmap1, fmap2)]
    for i in range(self.num_levels):
        fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
        fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
        pyramid.append((fmap1, fmap2))

    coords = coords.permute(0, 2, 3, 1)
    B, H, W, _ = coords.shape
    dim = pyramid[0][0].shape[1]

    corr_list = []
    for i in range(num_levels):
        fmap1_i = pyramid[0][0].permute(0, 2, 3, 1).contiguous()
        fmap2_i = pyramid[i][1].permute(0, 2, 3, 1).contiguous()

        coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
        corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, radius)
        corr_list.append(corr.squeeze(1))

    corr = torch.stack(corr_list, dim=1)
    corr = corr.reshape(B, -1, H, W)
    return corr / torch.sqrt(torch.tensor(dim).float())
