# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class GlobalAveragePoolingWithDropout(nn.Module):
    """Global Average Pooling neck with Dropout.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, kernel_size, dropout=None, dim=2):
        super(GlobalAveragePoolingWithDropout, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AvgPool1d(kernel_size)
        elif dim == 2:
            self.gap = nn.AvgPool2d(kernel_size)
        else:
            self.gap = nn.AvgPool3d(kernel_size)

        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            if self.dropout is not None:
                outs = tuple([self.dropout(x) for x in outs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            if self.dropout is not None:
                outs = self.dropout(outs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
