# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    L1MutableChannelUnit, MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.channel_unit import \
    ChannelUnit  # noqa
from mmrazor.structures.graph import ModuleGraph as ModuleGraph
from .....data.models import SingleLineModel
from .....data.tracer_passed_models import backward_passed_library

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


def _test_units(units: List[MutableChannelUnit], model):
    for unit in units:
        unit.prepare_for_pruning(model)
    mutable_units = [unit for unit in units if unit.is_mutable]
    assert len(mutable_units) >= 1, \
        'len of mutable units should greater or equal than 0.'
    for unit in mutable_units:
        choice = unit.sample_choice()
        unit.current_choice = choice
        assert abs(unit.current_choice - choice) < 0.1
    x = torch.rand([2, 3, 224, 224]).to(DEVICE)
    y = model(x)
    assert list(y.shape) == [2, 1000]


def _test_a_graph(model, graph):
    try:
        units = DefaultChannelUnit.init_from_graph(graph)
        _test_units(units, model)
        return True, ''
    except Exception as e:
        return False, f'{e},{graph}'


# def _test_a_model_from_fx_tracer(Model):
#     print(f'test {Model}')
#     model = Model()
#     model.eval()
#     model = model.to(DEVICE)
#     graph = ModuleGraph.init_from_fx_tracer(
#         model,
#         fx_tracer=dict(
#             type='RazorFxTracer',
#             is_extra_leaf_module=is_dynamic_op_for_fx_tracer,
#             concrete_args=dict(mode='tensor')))
#     return _test_a_graph(model, graph)

# def _test_a_model_from_backward_tracer(Model):
#     print(f'test {Model}')
#     model = Model()
#     model.eval()
#     model = model.to(DEVICE)
#     graph = ModuleGraph.init_from_backward_tracer(model)
#     return _test_a_graph(model, graph)


class TestMutableChannelUnit(TestCase):

    def test_init_from_graph(self):
        model = SingleLineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        units = DefaultChannelUnit.init_from_graph(graph)
        _test_units(units, model)

    def test_init_from_cfg(self):
        model = SingleLineModel()
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
        _test_units(units, model)

    def test_init_from_channel_unit(self):
        model = SingleLineModel()
        # init using tracer
        graph = ModuleGraph.init_from_backward_tracer(model)
        units: List[ChannelUnit] = ChannelUnit.init_from_graph(graph)
        mutable_units = [
            DefaultChannelUnit.init_from_channel_unit(unit) for unit in units
        ]
        _test_units(mutable_units, model)

    # def test_with_fx_tracer(self):
    #     test_models = FxPassedModelManager.include_models()
    #     with SetTorchThread(1):
    #         with mp.Pool() as p:
    #             result = p.map(_test_a_model_from_fx_tracer, test_models)
    #     for res, model_data in zip(result, test_models):
    #         with self.subTest(model=model_data):
    #             self.assertTrue(res[0], res[1])

    # def test_with_backward_tracer(self):
    #     test_models = BackwardPassedModelManager.include_models()
    #     with SetTorchThread(1):
    #         with mp.Pool() as p:
    #             result = p.map(_test_a_model_from_backward_tracer\
    # , test_models)
    #     for res, model_data in zip(result, test_models):
    #         with self.subTest(model=model_data):
    #             self.assertTrue(res[0], res[1])

    def test_replace_with_dynamic_ops(self):
        model_datas = backward_passed_library.include_models()
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
