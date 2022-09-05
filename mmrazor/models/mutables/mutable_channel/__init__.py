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
from .groups import (MUTABLECHANNELGROUP, DCFFChannelGroup, L1ChannelGroup,
                     MutableChannelGroup, OneShotChannelGroup,
                     SequentialChannelGroup, SlimmableChannelGroup)
from .mutable_channel_container import MutableChannelContainer
from .sequential_mutable_channel import SquentialMutableChannel
from .simple_mutable_channel import SimpleMutableChannel

__all__ = [
    'SimpleMutableChannel', 'L1ChannelGroup', 'SequentialChannelGroup',
    'MutableChannelGroup', 'OneShotChannelGroup', 'SlimmableChannelGroup',
    'BaseMutableChannel', 'MutableChannelContainer', 'SquentialMutableChannel',
    'MUTABLECHANNELGROUP', 'DCFFChannelGroup'
]
