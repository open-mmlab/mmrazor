# Copyright (c) OpenMMLab. All rights reserved.
from .mutable_channel_group import MutableChannelGroup
from .one_shot_channel_group import OneShotChannelGroup
from .simple_channel_group import SimpleChannelGroup
from .slimmable_channel_group import SlimmableChannelGroup

__all__ = [
    'MutableChannelGroup', 'SimpleChannelGroup', 'OneShotChannelGroup',
    'SlimmableChannelGroup'
]
