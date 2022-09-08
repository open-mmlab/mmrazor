# Copyright (c) OpenMMLab. All rights reserved.
r"""This module defines MutableChannels.

----------------------------------------------------------base_mutable_channel.py
BaseMutableChannel
|                            \
----------------------------------------------------------mutable_channel_container.py
MutableChannelContainer        \
----------------------------------------------------------other files
                                 \   other MutableChannels

MutableChannel are mainly used in DynamicOps. It helps DynamicOps to deal
with mutable number of channels.
"""
from .base_mutable_channel import BaseMutableChannel
from .groups import (ChannelGroupType, MutableChannelGroup,
                     SequentialChannelGroup)
from .mutable_channel import MutableChannel
from .mutable_channel_container import MutableChannelContainer
from .one_shot_mutable_channel import OneShotMutableChannel
from .sequential_mutable_channel import SquentialMutableChannel
from .simple_mutable_channel import SimpleMutableChannel
from .slimmable_mutable_channel import SlimmableMutableChannel

__all__ = [
    'SimpleMutableChannel', 'MutableChannelGroup', 'OneShotChannelGroup',
    'BaseMutableChannel', 'MutableChannelContainer', 'StackMutableChannel',
    'ChannelGroupType', 'MutableChannel', 'OneShotMutableChannel',
    'SlimmableMutableChannel', 'SquentialMutableChannel',
    'SequentialChannelGroup'
]
