# Copyright (c) OpenMMLab. All rights reserved.
from .channel_groups.mutable_channel_group import MutableChannelGroup
from .channel_groups.Simple_channel_group import SimpleChannelGroup
from .mutable_channel import MutableChannel
from .one_shot_mutable_channel import OneShotMutableChannel
from .simple_mutable_channel import SimpleMutableChannel
from .slimmable_mutable_channel import SlimmableMutableChannel

__all__ = [
    'OneShotMutableChannel', 'SlimmableMutableChannel', 'MutableChannel',
    'SimpleMutableChannel', 'SimpleChannelGroup', 'MutableChannelGroup'
]
