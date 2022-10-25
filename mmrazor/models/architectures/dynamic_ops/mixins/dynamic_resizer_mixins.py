# Copyright (c) OpenMMLab. All rights reserved.
from typing import Set

from torch import nn

from .dynamic_mixins import DynamicMixin


class DynamicResizeMixin(DynamicMixin):
    """A mixin class for Pytorch InputResizer, which can mutate ``shape``."""

    accepted_mutable_attrs: Set[str] = {'shape'}

    def register_mutable_attr(self, attr, mutable):
        if attr == 'shape':
            self._register_mutable_shape(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_shape(self, mutable_shape):
        assert hasattr(self, 'mutable_attrs')
        current_shape = mutable_shape.current_choice
        shape_dim = 1 if isinstance(current_shape, int) else len(current_shape)
        if shape_dim not in [1, 2, 3]:
            raise ValueError('Expect shape of mutable to be 1, 2 or 3'
                             f', but got: {shape_dim}.')

        self.mutable_attrs['shape'] = mutable_shape

    def get_dynamic_shape(self):
        if 'shape' in self.mutable_attrs:
            current_shape = self.mutable_attrs['shape'].current_choice
        else:
            current_shape = None
        return current_shape

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        if 'shape' in self.mutable_attrs:  # type:ignore
            fixed_shape = self.mutable_attrs[  # type:ignore
                'shape'].current_choice
        else:
            fixed_shape = None
        return self.static_op_factory(
            size=fixed_shape,
            interpolation_type=self._interpolation_type,  # type:ignore
            align_corners=self._align_corners,  # type:ignore
            scale_factor=self._scale_factor,  # type:ignore
            recompute_scale_factor=self._recompute_scale_factor)  # type:ignore
