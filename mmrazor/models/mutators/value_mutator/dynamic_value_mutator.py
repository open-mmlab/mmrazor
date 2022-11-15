# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.registry import MODELS
from ..group_mixin import DynamicSampleMixin
from .value_mutator import ValueMutator


@MODELS.register_module()
class DynamicValueMutator(ValueMutator, DynamicSampleMixin):

    @property
    def mutable_class_type(self):
        return OneShotMutableValue
