# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterator, Optional, Set

import torch.nn as nn
from mmengine.model import Sequential
from torch import Tensor
from torch.nn import Module

from mmrazor.models.mutables import DerivedMutable, MutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins import DynamicMixin


class DynamicSequential(Sequential, DynamicMixin):
    """Dynamic Sequential Container."""
    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs: Set[str] = {'depth'}

    forward_ignored_module = (MutableValue, DerivedMutable, nn.ModuleDict)

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        super().__init__(*args, init_cfg=init_cfg)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def mutable_depth(self):
        """Mutable depth."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['depth']

    def register_mutable_attr(self: Sequential, attr: str,
                              mutable: BaseMutable):
        """Register attribute of mutable."""
        if attr == 'depth':
            self._register_mutable_depth(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_depth(self: Sequential, mutable_depth: MutableValue):
        """Register mutable depth."""
        assert hasattr(self, 'mutable_attrs')
        assert mutable_depth.current_choice is not None
        current_depth = mutable_depth.current_choice
        if current_depth > len(self._modules):
            raise ValueError(f'Expect depth of mutable to be smaller than '
                             f'{len(self._modules)} as `depth`, '
                             f'but got: {current_depth}.')
        self.mutable_attrs['depth'] = mutable_depth

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return Sequential

    def to_static_op(self: Sequential) -> Sequential:
        """Convert dynamic Sequential to static one."""
        self.check_if_mutables_fixed()

        if self.mutable_depth is None:
            fixed_depth = len(self)
        else:
            fixed_depth = self.get_current_choice(self.mutable_depth)

        modules = []
        passed_module_nums = 0
        for module in self:
            if isinstance(module, self.forward_ignored_module):
                continue
            else:
                passed_module_nums += 1
            if passed_module_nums > fixed_depth:
                break

            modules.append(module)

        return Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Forward of Dynamic Sequential."""
        if self.mutable_depth is None:
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
