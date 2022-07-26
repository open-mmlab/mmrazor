# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.derived_mutable import DerivedMutable
from mmrazor.models.mutables.mutable_channel import MutableChannel
from mmrazor.registry import MODELS
from .base import MUTABLE_CFGS_TYPE, ChannelDynamicOP


class DynamicLinear(nn.Linear, ChannelDynamicOP):
    """Applies a linear transformation to the incoming data according to the
    `mutable_in_features` and `mutable_out_features` dynamically.

    Args:
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """

    accepted_mutable_keys = {'in_features', 'out_features'}

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **linear_kwargs) -> None:
        super().__init__(**linear_kwargs)

        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_features_mutable(mutable_cfgs)

    def _register_features_mutable(self,
                                   mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'in_features' or 'out_features' in mutable_cfgs:
            assert 'in_features' in mutable_cfgs and \
                'out_features' in mutable_cfgs, \
                'both `in_features` and `out_features` ' \
                'should be contained in `mutable_cfgs`'
            in_features = mutable_cfgs['in_features']
            if isinstance(in_features, dict):
                in_features.update(num_channels=self.in_features)
                in_features = MODELS.build(in_features)
            assert isinstance(in_features, (MutableChannel, DerivedMutable))
            self.in_features_mutable = in_features

            out_features = mutable_cfgs['out_features']
            if isinstance(out_features, dict):
                out_features.update(dict(num_channels=self.out_features))
                out_features = MODELS.build(out_features)
            assert isinstance(out_features, (MutableChannel, DerivedMutable))
            self.out_features_mutable = out_features

        else:
            self.register_parameter('in_features_mutable', None)
            self.register_parameter('out_features_mutable', None)

    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `in_features`."""
        return self.in_features_mutable

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `out_features`."""
        return self.out_features_mutable

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        in_mask = self.in_features_mutable.current_mask.to(self.weight.device)
        out_mask = self.out_features_mutable.current_mask.to(
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
