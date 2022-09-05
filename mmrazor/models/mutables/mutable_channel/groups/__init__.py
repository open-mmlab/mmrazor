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
from .l1_channel_group import L1ChannelGroup
from .mutable_channel_group import MUTABLECHANNELGROUP, MutableChannelGroup
from .one_shot_channel_group import OneShotChannelGroup
from .sequential_channel_group import SequentialChannelGroup
from .slimmable_channel_group import SlimmableChannelGroup

__all__ = [
    'L1ChannelGroup', 'MutableChannelGroup', 'SequentialChannelGroup',
    'OneShotChannelGroup', 'SlimmableChannelGroup', 'MUTABLECHANNELGROUP'
]
