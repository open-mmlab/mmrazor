# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ...ops import InputResizer
from ..mixins.dynamic_mixins import DynamicResizeMixin


@MODELS.register_module()
class DynamicInputResizer(InputResizer, DynamicResizeMixin):
    """Dynamic InputResizer Module.

    Note:
        Arguments for ``__init__`` of ``DynamicInputResizer`` is totally same
        as :obj:`mmrazor.models.architectures.InputResizer`.
    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `InputResizer`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """

    mutable_attrs: nn.ModuleDict

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    def forward(self,
                x: torch.Tensor,
                size=Optional[Tuple[int, int]]) -> torch.Tensor:
        self._size = self.get_dynamic_shape()

        if not self._size:
            self._size = size

        return super().forward(x, self._size)

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return InputResizer

    @classmethod
    def convert_from(cls, module: InputResizer):
        """Convert a InputResizer to a DynamicInputResizer.

        Args:
            module (:obj:`mmrazor.models.architectures.InputResizer`):
            The original InputResizer module.
        """
        dynamic_seq = cls(
            interpolation_type=module._interpolation_type,
            align_corners=module._align_corners,
            scale_factor=module._scale_factor)

        return dynamic_seq
