# Copyright (c) OpenMMLab. All rights reserved.
from .test_core.test_graph.test_graph import TestGraph  # isort:skip
from .test_core.test_graph.test_channel_graph import TestChannelGraph
from .test_core.test_tracer.test_backward_tracer import TestBackwardTracer
from .test_data import TestModelLibrary
from .test_models.test_algorithms.test_autoslim import TestAutoSlim
from .test_models.test_algorithms.test_prune_algorithm import \
    TestItePruneAlgorithm
from .test_models.test_algorithms.test_slimmable_network import (
    TestSlimmable, TestSlimmableDDP)
from .test_models.test_mutables.test_mutable_channel.test_units.test_mutable_channel_units import \
    TestMutableChannelUnit  # noqa: E501
from .test_models.test_mutators.test_channel_mutator import TestChannelMutator

__all__ = [
    'TestGraph', 'TestMutableChannelUnit', 'TestChannelMutator',
    'TestBackwardTracer', 'TestItePruneAlgorithm', 'TestAutoSlim',
    'TestSlimmable', 'TestSlimmableDDP', 'TestChannelGraph', 'TestModelLibrary'
]
