# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Type

import torch.nn as nn
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables import MutableChannelContainer
from .registry import MODELS


def replace_with_dynamic_ops(model: nn.Module,
                             dynamicop_map: Dict[Type[nn.Module],
                                                 Type[DynamicChannelMixin]]):
    """Replace torch modules with dynamic-ops."""

    def traverse_children(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, nn.Module):
                if type(module) in dynamicop_map:
                    new_module = dynamicop_map[type(module)].convert_from(
                        module)
                    setattr(model, name, new_module)
                else:
                    traverse_children(module)

    traverse_children(model)


def register_channel_container(model: nn.Module,
                               container_class: Type[MutableChannelContainer]):
    """register channel container for dynamic ops."""
    for module in model.modules():
        if isinstance(module, DynamicChannelMixin):
            in_channels = getattr(module, module.attr_mappings['in_channels'],
                                  0)
            if module.get_mutable_attr('in_channels') is None:
                module.register_mutable_attr('in_channels',
                                             container_class(in_channels))
            out_channels = getattr(module,
                                   module.attr_mappings['out_channels'], 0)
            if module.get_mutable_attr('out_channels') is None:

                module.register_mutable_attr('out_channels',
                                             container_class(out_channels))


# TO DO: Add more sub_model
# manage sub models for downstream repos
@MODELS.register_module()
def sub_model_prune(cfg, fix_subnet, prefix='', extra_prefix=''):
    model = MODELS.build(cfg)
    from mmrazor.structures import load_fix_subnet

    dynamicop_map = {
        Conv2dAdaptivePadding: dynamic_ops.DynamicConv2dAdaptivePadding,
        nn.Conv2d: dynamic_ops.DynamicConv2d,
        nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
        nn.Linear: dynamic_ops.DynamicLinear,
        nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
        EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
        _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
    }
    replace_with_dynamic_ops(model, dynamicop_map)
    register_channel_container(model, MutableChannelContainer)

    load_fix_subnet(
        model, fix_subnet, prefix=prefix, extra_prefix=extra_prefix)
    return model
