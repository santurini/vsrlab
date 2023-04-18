import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import correlation
from src.optical_flow.models.raft.utils import coords_grid, upflow

class RAFT(nn.Module):
    def __init__(
            self,
            small: bool = True,
            scale_factor: int = 4
    ):
        super().__init__()

        self.scale_factor = scale_factor

        if small:
            self.hidden_dim = 96
            self.context_dim = 64
            self.corr_levels = 4
            self.corr_radius = 3
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance')
            self.cnet = SmallEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='none')
            self.update_block = SmallUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim)
        
        else:
            self.hidden_dim = 128
            self.context_dim = 128
            self.corr_levels = 4
            self.corr_radius = 4
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
            self.cnet = BasicEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='batch')
            self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def initialize_flow(img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        return coords0, coords1

    def forward(self, image1, image2, iters=12):
        fmap1, fmap2 = self.fnet([image1, image2])

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = self.corr_fn(coords1, fmap1, fmap2, self.corr_levels, self.corr_radius)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

        flow_up = upflow(coords1-coords0, scale_factor=self.scale_factor)
            
        return flow_up
