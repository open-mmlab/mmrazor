# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_op.bricks.dynamic_mixins import \
    DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (MutableChannelGroup,
                                                     SimpleChannelGroup)
from mmrazor.models.mutables.mutable_channel.groups.channel_group import (  # noqa
    Channel, PruneNode, is_dynamic_op)
from mmrazor.structures.graph import ModuleGraph as ModuleGraph
from ...test_core.test_graph.test_graph import TestGraph

MUTABLE_CFG = dict(type='SimpleMutableChannl')
TRACER_CFG = dict(
    type='BackwardTracer',
    loss_calculator=dict(type='ImageClassifierPseudoLoss'))


def is_dynamic_op_for_tracer(module, module_name):
    """determine if a module is a dynamic op for fx tracer."""
    return is_dynamic_op(module)


# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() \
#     else torch.device('cpu')
DEVICE = torch.device('cpu')
GROUPS: List[MutableChannelGroup] = [SimpleChannelGroup]

DefaultChannelGroup = SimpleChannelGroup


class TestMutableChannelGroup(TestCase):

    def _test_a_graph(self, model, graph):
        try:
            groups = DefaultChannelGroup.parse_channel_groups(graph)
            for group in groups:
                group.prepare_for_pruning()
            prunable_groups = [group for group in groups if group.is_prunable]

            for group in prunable_groups:
                choice = group.sample_choice()
                group.current_choice = choice
                self.assertAlmostEqual(group.current_choice, choice, delta=0.1)
            x = torch.rand([2, 3, 224, 224]).to(DEVICE)
            y = model(x)
            self.assertSequenceEqual(y.shape, [2, 1000])

        except Exception as e:
            self.fail(f'{e}')

    def _test_a_model_using_backward_tracer(self, model):
        model.eval()
        model = model.to(DEVICE)
        graph = ModuleGraph.init_using_backward_tracer(model)
        self._test_a_graph(model, graph)

    def test_with_backward_tracer(self):
        test_models = TestGraph.backward_tracer_passed_models()
        for model_data in test_models:
            with self.subTest(model=model_data):
                model = model_data()
                self._test_a_model_using_backward_tracer(model)

                DefaultChannelGroup.prepare_model(model)
                self._test_a_model_using_backward_tracer(model)

    def test_group_split(self):
        layer = nn.Conv2d(3, 16, 3)
        node = PruneNode('layer', layer)
        channel1 = Channel(node, (8, 16), True)
        channel2 = Channel(node, (0, 8), True)
        group = DefaultChannelGroup(8)
        group.add_ouptut_related(channel1)
        group.add_ouptut_related(channel2)

        groups = group.split([2, 6])
        self.assertEqual(groups[0].output_related[0].index, (8, 10))
        self.assertEqual(groups[0].output_related[1].index, (0, 2))
        self.assertEqual(groups[1].output_related[0].index, (10, 16))
        self.assertEqual(groups[1].output_related[1].index, (2, 8))

    def test_replace_with_dynamic_ops(self):
        model_datas = TestGraph.backward_tracer_passed_models()[:1]
        for model_data in model_datas:
            for group_type in GROUPS:
                with self.subTest(model=model_data, group=group_type):
                    model: nn.Module = model_data()
                    group_type.prepare_model(model)

                    for module in model.modules():
                        if isinstance(module, nn.Conv2d):
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))
                        if isinstance(module, nn.Linear):
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))
                        if isinstance(module, nn.BatchNorm2d):
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))
