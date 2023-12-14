from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def iter_spatial_correlation_sample(
        input1: torch.Tensor,
        input2: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        patch_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        dilation_patch: Union[int, Tuple[int, int]] = 1
):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (dilation_patch, dilation_patch) if isinstance(dilation_patch, int) else dilation_patch

    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError('Only kernel_size=1 is supported.')
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError('Only dilation=1 is supported.')
    if (patch_size[0] % 2) == 0 or (patch_size[1] % 2) == 0:
        raise NotImplementedError('Only odd patch sizes are supperted.')

    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))

    max_displacement = (dilation_patch[0] * (patch_size[0] - 1) // 2, dilation_patch[1] * (patch_size[1] - 1) // 2)
    input2 = F.pad(input2, (max_displacement[1], max_displacement[1], max_displacement[0], max_displacement[0]))

    b, _, h, w = input1.shape
    input1 = input1[:, :, ::stride[0], ::stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw).to(dtype=input1.dtype, device=input1.device)

    for i in range(0, 2 * max_displacement[0] + 1, dilation_patch[0]):
        for j in range(0, 2 * max_displacement[1] + 1, dilation_patch[1]):
            p2 = input2[:, :, i:i + h, j:j + w]
            p2 = p2[:, :, ::stride[0], ::stride[1]]
            corr[:, i // dilation_patch[0], j // dilation_patch[1]] = (input1 * p2).sum(dim=1)

    return corr

class SpatialCorrelationSampler(nn.Module):

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]] = 1,
            patch_size: Union[int, Tuple[int, int]] = 1,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            dilation_patch: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(
            self,
            input1: torch.Tensor,
            input2: torch.Tensor
    ) -> torch.Tensor:
        return iter_spatial_correlation_sample(
            input1=input1, input2=input2, kernel_size=self.kernel_size, patch_size=self.patch_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation, dilation_patch=self.dilation_patch)
