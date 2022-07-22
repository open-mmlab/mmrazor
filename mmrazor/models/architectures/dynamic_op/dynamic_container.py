# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence

from mmengine.model import Sequential
from torch import Tensor, nn

from mmrazor.models.mutables.mutable_value import MutableValue
from mmrazor.registry import MODELS
from .base import MUTABLE_CFGS_TYPE, DynamicOP


class DynamicSequential(Sequential, DynamicOP):
    accepted_mutable_keys = {'length'}

    def __init__(self,
                 *,
                 modules: Sequence[nn.Module],
                 mutable_cfgs: MUTABLE_CFGS_TYPE,
                 init_cfg: Optional[dict] = None):
        super().__init__(*modules, init_cfg=init_cfg)
        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_length_mutable(mutable_cfgs)

    def _register_length_mutable(self,
                                 mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'length' in mutable_cfgs:
            length_cfg = copy.deepcopy(mutable_cfgs['length'])
            self.length_mutable = MODELS.build(length_cfg)
            assert isinstance(self.length_mutable, MutableValue)

            max_choice = self.length_mutable.max_choice
            # HACK
            # mutable will also be a submodule in sequential
            if len(self) - 1 != max_choice:
                raise ValueError('Max choice of length mutable must be the '
                                 'same as length of Sequential, but got max '
                                 f'choice: {max_choice}, expected max '
                                 f'length: {len(self)}.')
            self.length_mutable.current_choice = len(self) - 1
        else:
            self.register_parameter('length_mutable', None)

    def forward(self, x: Tensor) -> Tensor:
        current_length = self.length_mutable.current_choice

        passed_module_nums = 0
        for module in self:
            if not isinstance(module, MutableValue):
                passed_module_nums += 1
            if passed_module_nums > current_length:
                break

            x = module(x)

        return x

    def to_static_op(self) -> Sequential:
        self.check_if_mutables_fixed()

        fixed_length = self.length_mutable.current_choice

        modules = []
        passed_module_nums = 0
        for module in self:
            if not isinstance(module, MutableValue):
                passed_module_nums += 1
            if passed_module_nums > fixed_length:
                break

            modules.append(module)

        return Sequential(*modules)
