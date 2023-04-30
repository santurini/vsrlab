import logging

import torch
import torch.nn as nn
from optical_flow.models.irr.irr_modules import RefineFlow
from optical_flow.models.irr.pwc_modules import (
    conv, upsample2d_as, rescale_flow, initialize_msra, compute_cost_volume,
    WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense
)

from core import PROJECT_ROOT

pylogger = logging.getLogger(__name__)

class IRRPWCNet(nn.Module):
    def __init__(self, pretrained=False, return_levels=[-1, -2, -3, -4]):
        super().__init__()
        self._div_flow = 0.05
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.return_levels = return_levels
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in_flo = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in_flo)
        self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1)])

        self.conv_1x1_1 = conv(16, 3, kernel_size=1, stride=1, dilation=1)

        self.refine_flow = RefineFlow(2 + 1 + 32)
        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range,
                            "stride1": 1, "stride2": 1, "corr_multiply": 1}

        if pretrained:
            pylogger.info('Loading IRR pretrained weights')
            load_path = f'{PROJECT_ROOT}/src/optical_flow/weights/irr-sintel.ckpt'
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage), strict=True)

        else:
            initialize_msra(self.modules())

    def forward(self, ref, supp):

        x1_raw = supp
        x2_raw = ref
        batch_size, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows_f = [];
        flows_b = []

        _, _, h_x1, w_x1, = x1_pyramid[0].size()
        flow_f = torch.zeros(batch_size, 2, h_x1, w_x1).float().cuda()
        flow_b = torch.zeros(batch_size, 2, h_x1, w_x1).float().cuda()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            if l <= self.output_level:

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)

                # correlation
                out_corr_f = compute_cost_volume(x1, x2_warp, self.corr_params)
                out_corr_b = compute_cost_volume(x2, x1_warp, self.corr_params)

                out_corr_relu_f = self.leakyRELU(out_corr_f)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                if l != self.output_level:
                    x1_1by1 = self.conv_1x1[l](x1)
                    x2_1by1 = self.conv_1x1[l](x2)
                else:
                    x1_1by1 = x1
                    x2_1by1 = x2

                # concat and estimate flow
                flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

                x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
                x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
                flow_est_f = flow_f + flow_res_f
                flow_est_b = flow_b + flow_res_b

                flow_cont_f = flow_est_f + self.context_networks(torch.cat([x_intm_f, flow_est_f], dim=1))
                flow_cont_b = flow_est_b + self.context_networks(torch.cat([x_intm_b, flow_est_b], dim=1))

                # refinement
                img1_resize = upsample2d_as(x1_raw, flow_f, mode="bilinear")
                img2_resize = upsample2d_as(x2_raw, flow_b, mode="bilinear")
                img2_warp = self.warping_layer(img2_resize,
                                               rescale_flow(flow_cont_f, self._div_flow, width_im, height_im,
                                                            to_local=False), height_im, width_im, self._div_flow)
                img1_warp = self.warping_layer(img1_resize,
                                               rescale_flow(flow_cont_b, self._div_flow, width_im, height_im,
                                                            to_local=False), height_im, width_im, self._div_flow)

                # flow refine
                flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
                flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)

                flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

                flows_f.append(flow_f)
                flows_b.append(flow_b)

            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                flows_f.append(flow_f)
                flows_b.append(flow_b)

        flows_f = [flows_f[i] for i in self.return_levels]
        flows_b = [flows_b[i] for i in self.return_levels]

        return flows_f, flows_b
