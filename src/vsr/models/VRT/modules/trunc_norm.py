import math
import torch
import warnings

def norm_cdf(x):
    # Computes standard normal cumulative distribution function
    return (1. + math.erf(x / math.sqrt(2.))) / 2.

@torch.no_grad()
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    low = norm_cdf((a - mean) / std)
    up = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [low, up], then translate to [2l-1, 2u-1].
    tensor.uniform_(2 * low - 1, 2 * up - 1)

    # Use inverse cdf transform for normal distribution to get truncated standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)