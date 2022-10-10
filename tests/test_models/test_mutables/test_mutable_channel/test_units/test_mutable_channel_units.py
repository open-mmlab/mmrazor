# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    L1MutableChannelUnit, MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.channel_unit import (  # noqa
    Channel, ChannelUnit)
from mmrazor.structures.graph import ModuleGraph as ModuleGraph
from .....data.models import LineModel
from .....test_core.test_graph.test_graph import TestGraph

MUTABLE_CFG = dict(type='SimpleMutablechannel')
PARSE_CFG = dict(
    type='BackwardTracer',
    loss_calculator=dict(type='ImageClassifierPseudoLoss'))

# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() \
#     else torch.device('cpu')
DEVICE = torch.device('cpu')
GROUPS: List[MutableChannelUnit] = [
    L1MutableChannelUnit, SequentialMutableChannelUnit
]

DefaultChannelUnit = SequentialMutableChannelUnit


class TestMutableChannelUnit(TestCase):

    def test_init_from_graph(self):
        model = LineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        units = DefaultChannelUnit.init_from_graph(graph)
        self._test_units(units, model)

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
        units = [DefaultChannelUnit.init_from_cfg(model, config)]
        self._test_units(units, model)

    def test_init_from_channel_unit(self):
        model = LineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        units: List[ChannelUnit] = ChannelUnit.init_from_graph(graph)
        mutable_units = [
            DefaultChannelUnit.init_from_channel_unit(unit) for unit in units
        ]
        self._test_units(mutable_units, model)

    def _test_units(self, units: List[MutableChannelUnit], model):
        for unit in units:
            unit.prepare_for_pruning(model)
        mutable_units = [unit for unit in units if unit.is_mutable]
        self.assertGreaterEqual(len(mutable_units), 1)
        for unit in mutable_units:
            choice = unit.sample_choice()
            unit.current_choice = choice
            self.assertAlmostEqual(unit.current_choice, choice, delta=0.1)
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
            for unit_type in GROUPS:
                with self.subTest(model=model_data, unit=unit_type):
                    model: nn.Module = model_data()
                    graph = ModuleGraph.init_from_backward_tracer(model)
                    units: List[
                        MutableChannelUnit] = unit_type.init_from_graph(graph)
                    for unit in units:
                        unit.prepare_for_pruning(model)

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
            units = DefaultChannelUnit.init_from_graph(graph)
            self._test_units(units, model)
        except Exception as e:
            self.fail(f'{e}')
