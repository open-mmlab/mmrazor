# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


class DynamicLinear(nn.Linear, ChannelDynamicOP):
    """Applies a linear transformation to the incoming data according to the
    `mutable_in_features` and `mutable_out_features` dynamically.

    Args:
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """

    accepted_mutable_keys = {'in_features', 'out_features'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_features_mutable: Optional[BaseMutable] = None
        self.out_features_mutable: Optional[BaseMutable] = None

    def mutate_in_features(self, in_features_mutable: BaseMutable) -> None:
        self.check_channels_mutable(in_features_mutable)

        self.in_features_mutable = in_features_mutable

    def mutate_out_features(self, out_features_mutable: BaseMutable) -> None:
        self.check_channels_mutable(out_features_mutable)

        self.out_features_mutable = out_features_mutable

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable `in_features`."""
        return self.in_features_mutable

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable `out_features`."""
        return self.out_features_mutable

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self.in_features_mutable is None and \
                self.out_features_mutable is None:
            return self.weight, self.bias

        if self.in_features_mutable is not None:
            in_mask = self.in_features_mutable.current_mask.to(
                self.weight.device)
        else:
            in_mask = torch.ones(self.weight.size(1)).bool().to(
                self.weight.device)
        if self.out_features_mutable is not None:
            out_mask = self.out_features_mutable.current_mask.to(
                self.weight.device)
        else:
            out_mask = torch.ones(self.weight.size(0)).bool().to(
                self.weight.device)

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_features` and
        `mutable_out_features`, and forward."""
        weight, bias = self._get_dynamic_params()

        return F.linear(input, weight, bias)

    def to_static_op(self) -> nn.Module:
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
