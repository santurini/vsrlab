import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from core import PROJECT_ROOT
from optical_flow.models.raft.corr import correlation
from optical_flow.models.raft.extractor import BasicEncoder, SmallEncoder
from optical_flow.models.raft.update import BasicUpdateBlock, SmallUpdateBlock
from optical_flow.models.raft.utils import coords_grid, upflow

pylogger = logging.getLogger(__name__)

class RAFT(nn.Module):
    def __init__(
            self,
            small: bool = True,
            scale_factor: int = 2,
            pretrained: bool = True
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

            if pretrained:
                pylogger.info('Loading RAFT pretrained weights')
                state_dict = torch.load(f'{PROJECT_ROOT}/src/optical_flow/weights/raft-small.pth')
                new_dict = OrderedDict([(k.partition('module.')[-1], v) for k, v in state_dict.items()])
                self.load_state_dict(new_dict, strict=True)
        
        else:
            self.hidden_dim = 128
            self.context_dim = 128
            self.corr_levels = 4
            self.corr_radius = 4
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
            self.cnet = BasicEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='batch')
            self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim)

            if pretrained:
                pylogger.info('Loading <RAFT> pretrained weights')
                state_dict = torch.load(f'{PROJECT_ROOT}/src/optical_flow/weights/raft-sintel.pth')
                new_dict = OrderedDict([(k.partition('module.')[-1], v) for k, v in state_dict.items()])
                self.load_state_dict(new_dict)

    @staticmethod
    def initialize_flow(img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).type_as(img)
        coords1 = coords_grid(N, H//8, W//8).type_as(img)

        return coords0, coords1

    def forward(self, supp, ref, iters=12):
        supp = supp.contiguous()
        ref = ref.contiguous()

        fmap1, fmap2 = self.fnet([supp, ref])

        cnet = self.cnet(supp)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(supp)

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = correlation(coords1, fmap1, fmap2, self.corr_levels, self.corr_radius)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

        flow_up = upflow(coords1-coords0, scale_factor=self.scale_factor)

        print(torch.isnan(flow_up).sum())
            
        return flow_up
