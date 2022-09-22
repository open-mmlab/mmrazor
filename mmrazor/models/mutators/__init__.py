# Copyright (c) OpenMMLab. All rights reserved.
from .channel_mutator import (ChannelMutator, OneShotChannelMutator,
                              SlimmableChannelMutator)
from .module_mutator import (DiffModuleMutator, ModuleMutator,
                             OneShotModuleMutator)

__all__ = [
    'OneShotModuleMutator', 'DiffModuleMutator', 'ModuleMutator',
    'ChannelMutator', 'OneShotChannelMutator', 'SlimmableChannelMutator'
]
