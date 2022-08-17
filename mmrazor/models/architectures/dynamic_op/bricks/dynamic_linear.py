# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..base import ChannelDynamicOP


class DynamicLinear(nn.Linear, ChannelDynamicOP):
    """Dynamic Linear OP.

    Note:
        Arguments for ``__init__`` of ``DynamicLinear`` is totally same as
        :obj:`torch.nn.Linear`.

    Attributes:
        mutable_in_features (BaseMutable, optional): Mutable for controlling
            ``in_features``.
        mutable_out_features (BaseMutable, optional): Mutable for controlling
            ``out_features``.
    """
    accepted_mutable_attrs = {'in_features', 'out_features'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

        # self.mutable_in_features: Optional[BaseMutable] = None
        # self.mutable_out_features: Optional[BaseMutable] = None

    def forward(self, input: Tensor) -> Tensor:
        """Forward of dynamic linear OP."""
        weight, bias = self._get_dynamic_params()

        return F.linear(input, weight, bias)

    def to_static_op(self) -> nn.Module:
        """Convert dynamic linear to :obj:`torch.nn.Linear`.

        Returns:
            nn.Linear: :obj:`torch.nn.Linear` with sliced parameters.
        """
        self.check_if_mutables_fixed()

        weight, bias = self._get_dynamic_params()
        out_features = weight.size(0)
        in_features = weight.size(1)

        static_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True if bias is not None else False)

        static_linear.weight = nn.Parameter(weight)
        if bias is not None:
            static_linear.bias = nn.Parameter(bias)

        return static_linear
