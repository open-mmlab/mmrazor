# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence

from mmengine.model import Sequential
from torch import Tensor, nn

from mmrazor.models.mutables.mutable_value import MutableValue
from mmrazor.registry import MODELS
from .base import MUTABLE_CFGS_TYPE, DynamicOP


class DynamicSequential(Sequential, DynamicOP):
    accepted_mutable_keys = {'depth'}

    def __init__(self,
                 *,
                 modules: Sequence[nn.Module],
                 mutable_cfgs: MUTABLE_CFGS_TYPE,
                 init_cfg: Optional[dict] = None):
        super().__init__(*modules, init_cfg=init_cfg)
        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_depth_mutable(mutable_cfgs)

    def _register_depth_mutable(self, mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'depth' in mutable_cfgs:
            depth_cfg = copy.deepcopy(mutable_cfgs['depth'])
            self.depth_mutable = MODELS.build(depth_cfg)
            assert isinstance(self.depth_mutable, MutableValue)

            max_choice = self.depth_mutable.max_choice
            # HACK
            # mutable will also be a submodule in sequential
            if len(self) - 1 != max_choice:
                raise ValueError('Max choice of depth mutable must be the '
                                 'same as depth of Sequential, but got max '
                                 f'choice: {max_choice}, expected max '
                                 f'depth: {len(self)}.')
            self.depth_mutable.current_choice = len(self) - 1
        else:
            self.register_parameter('depth_mutable', None)

    def forward(self, x: Tensor) -> Tensor:
        current_depth = self.depth_mutable.current_choice

        passed_module_nums = 0
        for module in self:
            if not isinstance(module, MutableValue):
                passed_module_nums += 1
            if passed_module_nums > current_depth:
                break

            x = module(x)

        return x

    def to_static_op(self) -> Sequential:
        self.check_if_mutables_fixed()

        fixed_depth = self.depth_mutable.current_choice

        modules = []
        passed_module_nums = 0
        for module in self:
            if not isinstance(module, MutableValue):
                passed_module_nums += 1
            if passed_module_nums > fixed_depth:
                break

            modules.append(module)

        return Sequential(*modules)
