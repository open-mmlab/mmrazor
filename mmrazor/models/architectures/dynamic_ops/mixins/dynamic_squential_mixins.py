# Copyright (c) OpenMMLab. All rights reserved.
from typing import Set

from mmengine.model import Sequential
from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.mutables.mutable_value import MutableValue
from .dynamic_mixins import DynamicMixin


class DynamicSequentialMixin(DynamicMixin):

    accepted_mutable_attrs: Set[str] = {'depth'}

    @property
    def mutable_depth(self: Sequential) -> nn.Module:
        """Mutable depth."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['depth']
        # return self.mutable_attrs['depth'] if self.mutable_attrs.has_key('depth') else None

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
