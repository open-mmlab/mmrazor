# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
from torch import Tensor

from mmrazor.models.architectures.ops import RelativePosition2D
from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins.dynamic_rp_mixins import DynamicRelativePosition2DMixin

class DynamicRelativePosition2D(RelativePosition2D,
                                DynamicRelativePosition2DMixin):
    """Searchable RelativePosition module.

    Args:
        head_dims (int/Int): Parallel attention heads.
        head_dims ([int/Int]): embedding dims of relative position.
        max_relative_position ([int]): The max relative position distance.
    """
    accepted_mutable_attrs = {'head_dims'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return RelativePosition2D

    @classmethod
    def convert_from(cls, module):
        """Convert a RP to a dynamic RP."""
        dynamic_rp = cls(
            head_dims=module.head_dims,
            max_relative_position=module.max_relative_position)
        return dynamic_rp

    def forward(self, length_q, length_k) -> Tensor:
        """Forward of Dynamic Relative Position."""
        return self.forward_mixin(length_q, length_k)
