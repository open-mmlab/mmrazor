# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator, Optional, Sequence

from mmengine.model import Sequential
from torch import Tensor
from torch.nn import Module

from mmrazor.models.mutables import DerivedMutable, MutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import DynamicOP


class DynamicSequential(Sequential, DynamicOP):
    accpeted_mutables = {'mutable_depth'}
    forward_ignored_module = (MutableValue, DerivedMutable)

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        super().__init__(*args, init_cfg=init_cfg)

        self.mutable_depth: Optional[BaseMutable] = None

    def mutate_depth(self,
                     mutable_depth: BaseMutable,
                     depth_seq: Optional[Sequence[int]] = None) -> None:
        if depth_seq is None:
            depth_seq = getattr(mutable_depth, 'choices')
        if depth_seq is None:
            raise ValueError('depth sequence must be provided')
        depth_list = list(sorted(depth_seq))
        if depth_list[-1] != len(self):
            raise ValueError(f'Expect max depth to be: {len(self)}, '
                             f'but got: {depth_list[-1]}')

        self.depth_list = depth_list
        self.mutable_depth = mutable_depth

    def forward(self, x: Tensor) -> Tensor:
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
        nums = 0
        for _ in self.pure_modules():
            nums += 1

        return nums

    def pure_modules(self) -> Iterator[Module]:
        for module in self._modules.values():
            if isinstance(module, self.forward_ignored_module):
                continue
            yield module

    def to_static_op(self) -> Sequential:
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
