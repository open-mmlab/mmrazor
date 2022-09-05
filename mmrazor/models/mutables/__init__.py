# Copyright (c) OpenMMLab. All rights reserved.
from .base_mutable import BaseMutable
from .derived_mutable import DerivedMutable
from .mutable_channel import (BaseMutableChannel, MutableChannelContainer,
<<<<<<< HEAD
                              SimpleMutableChannel, StackMutableChannel)
from .mutable_channel.groups import (MUTABLECHANNELGROUP, DCFFChannelGroup,
                                     MutableChannelGroup, OneShotChannelGroup,
                                     SimpleChannelGroup, SlimmableChannelGroup)
=======
                              SimpleMutableChannel, SquentialMutableChannel)
from .mutable_channel.groups import (MUTABLECHANNELGROUP, L1ChannelGroup,
                                     MutableChannelGroup, OneShotChannelGroup,
                                     SequentialChannelGroup,
                                     SlimmableChannelGroup)
>>>>>>> lk
from .mutable_module import (DiffChoiceRoute, DiffMutableModule, DiffMutableOP,
                             OneShotMutableModule, OneShotMutableOP)
from .mutable_value import MutableValue, OneShotMutableValue

__all__ = [
    'OneShotMutableOP', 'OneShotMutableModule', 'DiffMutableOP',
    'DiffChoiceRoute', 'DiffMutableModule', 'DerivedMutable', 'MutableValue',
<<<<<<< HEAD
    'OneShotMutableValue', 'SimpleChannelGroup', 'OneShotChannelGroup',
    'SimpleMutableChannel', 'MutableChannelGroup', 'SlimmableChannelGroup',
    'BaseMutableChannel', 'MutableChannelContainer', 'MUTABLECHANNELGROUP',
    'StackMutableChannel', 'BaseMutable', 'DCFFChannelGroup'
=======
    'OneShotMutableValue', 'SequentialChannelGroup', 'L1ChannelGroup',
    'OneShotChannelGroup', 'SimpleMutableChannel', 'MutableChannelGroup',
    'SlimmableChannelGroup', 'BaseMutableChannel', 'MutableChannelContainer',
    'MUTABLECHANNELGROUP', 'SquentialMutableChannel', 'BaseMutable'
>>>>>>> lk
]
