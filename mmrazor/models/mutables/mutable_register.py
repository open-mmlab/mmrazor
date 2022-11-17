# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

from mmrazor.models.architectures.ops.gml_mobilenet_series import (GMLMBBlock,
                                                                   GMLSELayer)
from mmrazor.models.mutables.mutable_value import MutableValue
from .base_mutable import BaseMutable
from .derived_mutable import DerivedMutable
from .mutable_channel import MutableChannelContainer


def mutate_conv_module(
        conv_module,
        mutable_in_channels: Optional[BaseMutable] = None,
        mutable_out_channels: Optional[BaseMutable] = None,
        mutable_kernel_size: Optional[Tuple[BaseMutable,
                                            Sequence[int]]] = None):
    """Mutate a conv module."""

    def get_end_channels(mutable_num):
        """Get end channels based on ratio & mutable_num's current_choice."""
        if isinstance(mutable_num, DerivedMutable):
            source_mutables = mutable_num._trace_source_mutables()
            source_value_mutables = [
                mutable for mutable in source_mutables
                if isinstance(mutable, MutableValue)
            ]
            ratio = int(source_value_mutables[0].current_choice) \
                if len(source_value_mutables) > 0 else 1
            end_channels = mutable_num.current_choice * ratio
        else:
            end_channels = -1
        return end_channels

    if mutable_in_channels is not None:
        MutableChannelContainer.register_mutable_channel_to_module(
            conv_module.conv,
            mutable_in_channels,
            False,
            end=get_end_channels(mutable_in_channels))

    if mutable_out_channels is not None:
        MutableChannelContainer.register_mutable_channel_to_module(
            conv_module.conv,
            mutable_out_channels,
            True,
            end=get_end_channels(mutable_out_channels))

        if hasattr(conv_module, 'bn'):
            MutableChannelContainer.register_mutable_channel_to_module(
                conv_module.bn,
                mutable_out_channels,
                False,
                end=get_end_channels(mutable_out_channels))

    if mutable_kernel_size is not None:
        conv_module.conv.register_mutable_attr('kernel_size',
                                               mutable_kernel_size)


def mutate_se_layer(se_layer: GMLSELayer, mutable_in_channels: BaseMutable,
                    se_cfg: dict):
    # TODO: make divisiable
    ratio = se_cfg.get('ratio', 16)
    divisor = se_cfg.get('divisor', 8)
    derived_mid_channels = mutable_in_channels.derive_divide_mutable(
        ratio, divisor)
    mutate_conv_module(
        se_layer.conv1,
        mutable_in_channels=mutable_in_channels,
        mutable_out_channels=derived_mid_channels)
    mutate_conv_module(
        se_layer.conv2,
        mutable_in_channels=derived_mid_channels,
        mutable_out_channels=mutable_in_channels)


def mutate_mobilenet_layer(mb_layer: GMLMBBlock, mutable_in_channels,
                           mutable_out_channels, mutable_expand_value,
                           mutable_kernel_size, se_cfg):
    """Mutate mobilenet layers."""
    derived_expand_channels = mutable_expand_value * mutable_in_channels

    if mb_layer.with_expand_conv:
        mutate_conv_module(
            mb_layer.expand_conv,
            mutable_in_channels=mutable_in_channels,
            mutable_out_channels=derived_expand_channels)

    mutate_conv_module(
        mb_layer.depthwise_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=derived_expand_channels,
        mutable_kernel_size=mutable_kernel_size)

    if mb_layer.with_se:
        mutate_se_layer(
            mb_layer.se,
            mutable_in_channels=derived_expand_channels,
            se_cfg=se_cfg)

    if not mb_layer.with_res_shortcut:
        if mb_layer.with_attentive_shortcut:
            MutableChannelContainer.register_mutable_channel_to_module(
                mb_layer.shortcut.conv, mutable_in_channels, False)
            MutableChannelContainer.register_mutable_channel_to_module(
                mb_layer.shortcut.conv, mutable_out_channels, True)

    mutate_conv_module(
        mb_layer.linear_conv,
        mutable_in_channels=derived_expand_channels,
        mutable_out_channels=mutable_out_channels)
