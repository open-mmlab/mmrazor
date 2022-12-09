# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmrazor.registry import MODELS


@MODELS.register_module()
class SqueezeMeanPoolingWithDropout(BaseModule):
    """Dimensionality Reduction Neck with Dropout.

    Dimensionality Reduction the feature map of backbone by SqueezeMean.
    Some of the code is borrowed from
        `https://github.com/facebookresearch/AttentiveNAS`.

    Args:
        drop_ratio (float): Dropout rate. Defaults to 0.2.
    """

    def __init__(self, drop_ratio: float = 0.2):
        super(SqueezeMeanPoolingWithDropout, self).__init__()
        self.drop_ratio = drop_ratio

    def dimension_reduction(self, x: torch.Tensor):
        assert x.ndim > 1, 'SqueezeMean only support (B, C, *) input.'
        'to B C*H*W output if dim = 2'
        for i in range(x.ndim - 1, 1, -1):
            x = x.mean(i, keepdim=True)
            x = torch.squeeze(x, -1)
        return x

    def forward(
            self, inputs: Union[Tuple,
                                torch.Tensor]) -> Union[Tuple, torch.Tensor]:
        """Forward function with dropout.

        Args:
            x (Union[Tuple, torch.Tensor]): The feature map of backbone.
        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        drop_ratio = self.drop_ratio if self.drop_ratio is not None else 0.0

        if isinstance(inputs, tuple):
            outs = tuple([self.dimension_reduction(x) for x in inputs])
            if drop_ratio > 0 and self.training:
                outs = tuple([F.dropout(x, p=drop_ratio) for x in outs])
        elif isinstance(inputs, torch.Tensor):
            inputs = self.dimension_reduction(inputs)
            if drop_ratio > 0 and self.training:
                outs = F.dropout(inputs, p=drop_ratio)  # type:ignore
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
