# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.impl.pruning.group_fisher.algorithm import GroupFisherAlgorithm
from .dcff import DCFF
from .slimmable_network import SlimmableNetwork, SlimmableNetworkDDP

__all__ = [
    'SlimmableNetwork', 'SlimmableNetworkDDP', 'DCFF', 'GroupFisherAlgorithm'
]
