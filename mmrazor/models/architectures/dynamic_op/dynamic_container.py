# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.model import Sequential
from torch import Tensor

from mmrazor.models.mutables import DerivedMutable, MutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import DynamicOP


class DynamicSequential(Sequential, DynamicOP):
    accepted_mutable_keys = {'depth'}
    forward_ignored_module = (MutableValue, DerivedMutable)

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        super().__init__(*args, init_cfg=init_cfg)

        self.depth_mutable: Optional[BaseMutable] = None

    def mutate_depth(self,
                     depth_mutable: BaseMutable,
                     depth_seq: Optional[Sequence[int]] = None) -> None:
        if depth_seq is None:
            depth_seq = getattr(depth_mutable, 'choices')
        if depth_seq is None:
            raise ValueError('depth sequence must be provided')
        depth_list = list(sorted(depth_seq))
        if depth_list[-1] != len(self):
            raise ValueError(f'Expect max depth to be: {len(self)}, '
                             f'but got: {depth_list[-1]}')

        self.depth_list = depth_list
        self.depth_mutable = depth_mutable

    def forward(self, x: Tensor) -> Tensor:
        if self.depth_mutable is None:
            return self(x)

        current_depth = self.get_current_choice(self.depth_mutable)
        passed_module_nums = 0
        for module in self:
            if not isinstance(module, self.forward_ignored_module):
                passed_module_nums += 1
            if passed_module_nums > current_depth:
                break

            x = module(x)

        return x

    def to_static_op(self) -> Sequential:
        self.check_if_mutables_fixed()

        if self.depth_mutable is None:
            fixed_depth = len(self)
        else:
            fixed_depth = self.get_current_choice(self.depth_mutable)

        modules = []
        passed_module_nums = 0
        for module in self:
            if not isinstance(module, self.forward_ignored_module):
                passed_module_nums += 1
            if passed_module_nums > fixed_depth:
                break

            modules.append(module)

        return Sequential(*modules)
