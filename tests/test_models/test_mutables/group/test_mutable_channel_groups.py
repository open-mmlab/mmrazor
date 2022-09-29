# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    L1MutableChannelGroup, MutableChannelGroup, SequentialMutableChannelGroup)
from mmrazor.models.mutables.mutable_channel.units.channel_group import (  # noqa
    Channel, ChannelGroup)
from mmrazor.structures.graph import ModuleGraph as ModuleGraph
from ....data.models import LineModel
from ....test_core.test_graph.test_graph import TestGraph

MUTABLE_CFG = dict(type='SimpleMutablechannel')
PARSE_CFG = dict(
    type='BackwardTracer',
    loss_calculator=dict(type='ImageClassifierPseudoLoss'))

# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() \
#     else torch.device('cpu')
DEVICE = torch.device('cpu')
GROUPS: List[MutableChannelGroup] = [
    L1MutableChannelGroup, SequentialMutableChannelGroup
]

DefaultChannelGroup = SequentialMutableChannelGroup


class TestMutableChannelGroup(TestCase):

    def test_init_from_graph(self):
        model = LineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        groups = DefaultChannelGroup.init_from_graph(graph)
        self._test_groups(groups, model)

    def test_init_from_cfg(self):
        model = LineModel()
        # init using tracer

        config = {
            'init_args': {
                'num_channels': 8
            },
            'channels': {
                'input_related': [{
                    'name': 'net.1',
                    'start': 0,
                    'end': 8,
                    'expand_ratio': 1,
                    'is_output_channel': False
                }, {
                    'name': 'net.3',
                    'start': 0,
                    'end': 8,
                    'expand_ratio': 1,
                    'is_output_channel': False
                }],
                'output_related': [{
                    'name': 'net.0',
                    'start': 0,
                    'end': 8,
                    'expand_ratio': 1,
                    'is_output_channel': True
                }, {
                    'name': 'net.1',
                    'start': 0,
                    'end': 8,
                    'expand_ratio': 1,
                    'is_output_channel': True
                }]
            }
        }
        groups = [DefaultChannelGroup.init_from_cfg(model, config)]
        self._test_groups(groups, model)

    def test_init_from_channel_group(self):
        model = LineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        groups: List[ChannelGroup] = ChannelGroup.init_from_graph(graph)
        mutable_groups = [
            DefaultChannelGroup.init_from_channel_group(group)
            for group in groups
        ]
        self._test_groups(mutable_groups, model)

    def _test_groups(self, groups: List[MutableChannelGroup], model):
        for group in groups:
            group.prepare_for_pruning(model)
        mutable_groups = [group for group in groups if group.is_mutable]
        self.assertGreaterEqual(len(mutable_groups), 1)
        for group in mutable_groups:
            choice = group.sample_choice()
            group.current_choice = choice
            self.assertAlmostEqual(group.current_choice, choice, delta=0.1)
        x = torch.rand([2, 3, 224, 224]).to(DEVICE)
        y = model(x)
        self.assertSequenceEqual(y.shape, [2, 1000])

    def _test_a_model_from_backward_tracer(self, model):
        model.eval()
        model = model.to(DEVICE)
        graph = ModuleGraph.init_from_backward_tracer(model)
        self._test_a_graph(model, graph)

    def test_with_backward_tracer(self):
        test_models = TestGraph.backward_tracer_passed_models()
        for model_data in test_models:
            with self.subTest(model=model_data):
                model = model_data()
                self._test_a_model_from_backward_tracer(model)

    def test_replace_with_dynamic_ops(self):
        model_datas = TestGraph.backward_tracer_passed_models()
        for model_data in model_datas:
            for group_type in GROUPS:
                with self.subTest(model=model_data, group=group_type):
                    model: nn.Module = model_data()
                    graph = ModuleGraph.init_from_backward_tracer(model)
                    groups: List[
                        MutableChannelGroup] = group_type.init_from_graph(
                            graph)

                    for group in groups:
                        group.prepare_for_pruning(model)

                    for module in model.modules():
                        if isinstance(module, nn.Conv2d)\
                            and module.groups == module.in_channels\
                                and module.groups == 1:
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))
                        if isinstance(module, nn.Linear):
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))
                        if isinstance(module, nn.BatchNorm2d):
                            self.assertTrue(
                                isinstance(module, DynamicChannelMixin))

    def _test_a_graph(self, model, graph):
        try:
            groups = DefaultChannelGroup.init_from_graph(graph)
            self._test_groups(groups, model)
        except Exception as e:
            self.fail(f'{e}')
