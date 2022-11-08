# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import (
    DynamicChannelMixin, DynamicMHAMixin)
from mmrazor.models.mutables.mutable_channel import MutableChannelContainer
from mmrazor.registry import MODELS
from .one_shot_mutable_channel_unit import OneShotMutableChannelUnit


@MODELS.register_module()
class OneShotMutableChannelUnit_VIT(OneShotMutableChannelUnit):

    MixinScope = {
        'naive': (DynamicChannelMixin),
        'mix': (DynamicMHAMixin, DynamicChannelMixin)
    }

    def prepare_for_pruning(self,
                            model: nn.Module,
                            unit_predefined: bool = True):
        """Prepare for pruning."""
        if not unit_predefined:
            super().prepare_for_pruning(model)
        self.current_choice = self.max_choice

    @staticmethod
    def _register_channel_container(
            model: nn.Module,
            container_class: Type[MutableChannelContainer],
            extra_mixin: str = 'mix'):
        """register channel container for dynamic ops."""
        for module in model.modules():
            if isinstance(
                    module,
                    OneShotMutableChannelUnit_VIT.MixinScope[extra_mixin]):
                if module.get_mutable_attr('in_channels') is None:
                    in_channels = module.in_channels
                    module.register_mutable_attr('in_channels',
                                                 container_class(in_channels))
                if module.get_mutable_attr('out_channels') is None:
                    out_channels = module.out_channels
                    module.register_mutable_attr('out_channels',
                                                 container_class(out_channels))
