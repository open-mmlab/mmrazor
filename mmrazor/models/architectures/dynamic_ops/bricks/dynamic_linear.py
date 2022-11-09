# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins import DynamicLinearMixin


class DynamicLinear(nn.Linear, DynamicLinearMixin):
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

    @property
    def static_op_factory(self):
        return nn.Linear

    @classmethod
    def convert_from(cls, module):
        """Convert a nn.Linear module to a DynamicLinear.

        Args:
            module (:obj:`torch.nn.Linear`): The original Linear module.
        """
        dynamic_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False)
        return dynamic_linear

    def forward(self, input: Tensor) -> Tensor:
        """Forward of dynamic linear OP."""
        weight, bias = self.get_dynamic_params()

        return F.linear(input, weight, bias)
