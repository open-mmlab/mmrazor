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
from .dcff_channel_group import DCFFChannelGroup
from .mutable_channel_group import ChannelGroupType, MutableChannelGroup
from .one_shot_mutable_channel_group import OneShotMutableChannelGroup
from .sequential_mutable_channel_group import SequentialMutableChannelGroup

__all__ = [
    'MutableChannelGroup', 'SequentialMutableChannelGroup',
    'OneShotMutableChannelGroup', 'ChannelGroupType', 'DCFFChannelGroup'
]
