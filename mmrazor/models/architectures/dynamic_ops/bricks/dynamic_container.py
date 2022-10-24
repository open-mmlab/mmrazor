# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterator, Optional

import torch.nn as nn
from mmengine.model import Sequential
from torch import Tensor
from torch.nn import Module

from mmrazor.models.mutables import DerivedMutable, MutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins.dynamic_squential_mixins import DynamicSequentialMixin


class DynamicSequential(Sequential, DynamicSequentialMixin):
    """Dynamic Sequential Container."""
    accepted_mutable_attrs = {'depth'}

    forward_ignored_module = (MutableValue, DerivedMutable, nn.ModuleDict)

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        super().__init__(*args, init_cfg=init_cfg)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def forward(self, x: Tensor) -> Tensor:
        """Forward of Dynamic Sequential."""
        if self.mutable_depth is None:
        # if not hasattr(self, 'mutable_depth'):
            return self(x)

        current_depth = self.get_current_choice(self.mutable_depth)
        passed_module_nums = 0
        for module in self.pure_modules():
            passed_module_nums += 1
            if passed_module_nums > current_depth:
                break
            x = module(x)
        return x

    @property
    def pure_module_nums(self) -> int:
        """Number of pure module."""
        return sum(1 for _ in self.pure_modules())

    def pure_modules(self) -> Iterator[Module]:
        """nn.Module would influence the forward of Sequential."""
        for module in self._modules.values():
            if isinstance(module, self.forward_ignored_module):
                continue
            yield module

    @classmethod
    def convert_from(cls, module: Sequential):
        """Convert the static Sequential to dynamic one."""
        dynamic_m = cls(module._modules)
        return dynamic_m
