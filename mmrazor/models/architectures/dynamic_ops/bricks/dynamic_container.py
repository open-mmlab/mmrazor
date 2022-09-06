# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch.nn as nn
from mmengine.model import Sequential
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from .dynamic_mixins import DynamicSequentialMixin


@MODELS.register_module()
class DynamicSequential(Sequential, DynamicSequentialMixin):
    """Dynamic Sequential OP.

    Note:
        Arguments for ``__init__`` of ``DynamicSequential`` is totally same as
        :obj:`torch.nn.Sequential`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `depth`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    def forward(self, x: Tensor) -> Tensor:
        current_depth = self.get_dynamic_depth()

        for idx, module in enumerate(self):
            if idx >= current_depth:
                break
            x = module(x)
        return x

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return Sequential

    @classmethod
    def convert_from(cls, module: Sequential):
        """Convert a Sequential module to a DynamicSequential.

        Args:
            module (:obj:`torch.nn.Sequential`): The original Sequential
                module.
        """
        dynamic_seq = cls(**module._modules())

        return dynamic_seq
