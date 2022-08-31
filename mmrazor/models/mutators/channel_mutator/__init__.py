# Copyright (c) OpenMMLab. All rights reserved.
from .bignas_channel_mutator import BigNASChannelMutator
from .channel_mutator import ChannelMutator
from .one_shot_channel_mutator import OneShotChannelMutator
from .slimmable_channel_mutator import SlimmableChannelMutator

__all__ = [
    'ChannelMutator', 'OneShotChannelMutator', 'SlimmableChannelMutator',
    'BigNASChannelMutator'
]
