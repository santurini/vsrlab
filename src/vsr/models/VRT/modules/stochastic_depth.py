import torch

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape).type_as(x)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output