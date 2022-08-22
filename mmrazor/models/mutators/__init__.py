# Copyright (c) OpenMMLab. All rights reserved.
from .channel_mutator import (ChannelMutator, OneShotChannelMutator,
                              SlimmableChannelMutator)
from .channel_mutator.channel_group_mutator import ChannelGroupMutator
from .module_mutator import (DiffModuleMutator, ModuleMutator,
                             OneShotModuleMutator)

__all__ = [
    'OneShotModuleMutator', 'DiffModuleMutator', 'ModuleMutator',
    'ChannelMutator', 'OneShotChannelMutator', 'SlimmableChannelMutator',
    'ChannelGroupMutator'
]
