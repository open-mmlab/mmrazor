# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch.nn import Module

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.ops import InputResizer
from mmrazor.registry import MODELS
from .base import DynamicOP


# TODO
# consider use data preprocessor
@MODELS.register_module()
class DynamicInputResizer(InputResizer, DynamicOP):
    valid_interpolation_type = {
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area',
        'nearest-exact'
    }
    accpeted_mutables = {'mutable_shape'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_shape: Optional[BaseMutable] = None

    def mutate_shape(self, mutable_shape: BaseMutable) -> None:
        self.mutable_shape = mutable_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mutable_shape is not None:
            self._size = self.mutable_shape.current_choice

        return super().forward(x)

    def to_static_op(self) -> Module:
        self.check_if_mutables_fixed()

        size = None
        if self.mutable_shape is not None:
            size = self.get_current_choice(self.mutable_shape)
        return InputResizer(
            size=size,
            interpolation_type=self._interpolation_type,
            align_corners=self._align_corners,
            scale_factor=self._scale_factor,
            recompute_scale_factor=self._recompute_scale_factor)
