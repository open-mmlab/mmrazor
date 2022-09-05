# Copyright (c) OpenMMLab. All rights reserved.
from .test_core.test_graph.test_graph import TestGraph
from .test_core.test_tracer.test_backward_tracer import TestBackwardTracer
from .test_models.test_algorithms.test_autoslim import TestAutoSlim
from .test_models.test_algorithms.test_prune_algorithm import \
    TestItePruneAlgorithm
from .test_models.test_algorithms.test_slimmable_network import TestSlimmable
from .test_models.test_mutables.group.test_mutable_channel_groups import \
    TestMutableChannelGroup
from .test_models.test_mutators.test_channel_mutator import TestChannelMutator

__all__ = [
    'TestGraph', 'TestMutableChannelGroup', 'TestNode', 'TestChannelMutator',
    'TestBackwardTracer', 'TestItePruneAlgorithm', 'TestAutoSlim',
    'TestSlimmable'
]
