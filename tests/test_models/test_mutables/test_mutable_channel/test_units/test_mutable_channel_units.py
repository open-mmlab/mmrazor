# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    L1MutableChannelUnit, MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.channel_unit import \
    ChannelUnit
from .....data.models import SingleLineModel
from .....data.tracer_passed_models import backward_passed_library

MUTABLE_CFG = dict(type='SimpleMutablechannel')
PARSE_CFG = dict(
    type='ChannelAnalyzer',
    demo_input=(1, 3, 224, 224),
    tracer_type='BackwardTracer')

DEVICE = torch.device('cpu')
UNITS: List[MutableChannelUnit] = [
    L1MutableChannelUnit, SequentialMutableChannelUnit
]

DefaultChannelUnit = SequentialMutableChannelUnit


def _test_units(units: List[MutableChannelUnit], model):
    for unit in units:
        unit.prepare_for_pruning(model)
    for unit in units:
        _ = unit.current_choice

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


class TestMutableChannelUnit(TestCase):

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

    def test_init(self):
        for UnitClass in UNITS:
            with self.subTest(unit_class=UnitClass):

                def test_units(units, model):
                    mutable_units = [
                        UnitClass.init_from_channel_unit(unit)
                        for unit in units
                    ]
                    _test_units(mutable_units, model)

                # init using tracer
                model = SingleLineModel()
                units: List[
                    ChannelUnit] = ChannelUnit.init_from_channel_analyzer(
                        model)
                test_units(units, model)

                # init using tracer config
                model = SingleLineModel()
                units: List[
                    ChannelUnit] = ChannelUnit.init_from_channel_analyzer(
                        model, analyzer=dict(type='ChannelAnalyzer'))
                test_units(units, model)

                print(units)

    def test_replace_with_dynamic_ops(self):
        model_datas = backward_passed_library.include_models()
        for model_data in model_datas:
            for unit_type in UNITS:
                with self.subTest(model=model_data, unit=unit_type):
                    model: nn.Module = model_data()
                    units: List[
                        MutableChannelUnit] = unit_type.init_from_channel_analyzer(  # noqa
                            model)
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
