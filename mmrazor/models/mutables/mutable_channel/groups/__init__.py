# Copyright (c) OpenMMLab. All rights reserved.
"""
----------------------------------------------------------channel_group.py
PruneNode && PruneGraph
|
| Graph2ChannelGroups
|
Channel && ChannelGroup
|
----------------------------------------------------------mutable_channel_group.py
MutableChannelGroup
|
----------------------------------------------------------other files
Subclasses of MutableChannelGroup
"""
from .mutable_channel_group import ChannelGroupType, MutableChannelGroup
from .sequential_channel_group import SequentialChannelGroup

__all__ = ['MutableChannelGroup', 'SequentialChannelGroup', 'ChannelGroupType']
