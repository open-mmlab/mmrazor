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
from .l1_mutable_channel_group import L1MutableChannelGroup
from .mutable_channel_group import ChannelGroupType, MutableChannelGroup
from .one_shot_mutable_channel_group import OneShotMutableChannelGroup
from .sequential_mutable_channel_group import SequentialMutableChannelGroup
from .slimmable_channel_group import SlimmableChannelGroup

__all__ = [
    'L1MutableChannelGroup', 'MutableChannelGroup',
    'SequentialMutableChannelGroup', 'OneShotMutableChannelGroup',
    'SlimmableChannelGroup', 'ChannelGroupType'
]
