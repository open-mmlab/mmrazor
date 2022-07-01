# Copyright (c) OpenMMLab. All rights reserved.
from .channel_mutator.one_shot_channel_mutator import OneShotChannelMutator
from .channel_mutator.slimmable_channel_mutator import SlimmableChannelMutator
from .diff_mutator import DiffMutator
from .one_shot_mutator import OneShotMutator

__all__ = [
    'OneShotMutator', 'OneShotChannelMutator', 'SlimmableChannelMutator',
    'DiffMutator'
]
