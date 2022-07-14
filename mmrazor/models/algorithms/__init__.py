# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseAlgorithm
from .distill import FpnTeacherDistill, SingleTeacherDistill
from .nas import SPOS, AutoSlim, AutoSlimDDP
from .pruning import SlimmableNetwork, SlimmableNetworkDDP

__all__ = [
    'SingleTeacherDistill', 'BaseAlgorithm', 'FpnTeacherDistill', 'SPOS',
    'SlimmableNetwork', 'SlimmableNetworkDDP', 'AutoSlim', 'AutoSlimDDP'
]
