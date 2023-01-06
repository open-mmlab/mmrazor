# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.registry import MODELS
from .value_mutator import ValueMutator


@MODELS.register_module()
class DynamicValueMutator(ValueMutator):
    """Dynamic value mutator with type as `OneShotMutableValue`."""

    @property
    def mutable_class_type(self):
        return OneShotMutableValue
