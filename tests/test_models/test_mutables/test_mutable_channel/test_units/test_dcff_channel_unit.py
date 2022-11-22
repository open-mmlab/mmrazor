# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops import FuseConv2d
from mmrazor.models.mutables import DCFFChannelUnit
from mmrazor.structures.graph import ModuleGraph as ModuleGraph
from .....data.models import LineModel

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() \
    else torch.device('cpu')


class TestDCFFChannelUnit(TestCase):

    def test_num(self):
        unit = DCFFChannelUnit(48, choice_mode='number')
        unit.current_choice = 24
        self.assertEqual(unit.current_choice, 24)

        unit.current_choice = 0.5
        self.assertEqual(unit.current_choice, 24)

    def test_ratio(self):
        unit = DCFFChannelUnit(48, choice_mode='ratio')
        unit.current_choice = 0.5
        self.assertEqual(unit.current_choice, 0.5)
        unit.current_choice = 24
        self.assertEqual(unit.current_choice, 0.5)

    def test_divisor(self):
        unit = DCFFChannelUnit(48, choice_mode='number', divisor=8)
        unit.current_choice = 20
        self.assertEqual(unit.current_choice, 24)
        self.assertTrue(unit.sample_choice() % 8 == 0)

        unit = DCFFChannelUnit(48, choice_mode='ratio', divisor=8)
        unit.current_choice = 0.3
        self.assertEqual(unit.current_choice, 1 / 3)

    def test_config_template(self):
        unit = DCFFChannelUnit(48, choice_mode='ratio', divisor=8)
        config = unit.config_template(with_init_args=True)
        unit2 = DCFFChannelUnit.init_from_cfg(None, config)
        self.assertDictEqual(
            unit2.config_template(with_init_args=True)['init_args'],
            config['init_args'])

    def test_init_from_channel_unit(self):
        # init using tracer
        model = LineModel()
        graph = ModuleGraph.init_from_backward_tracer(model)
        units: List[DCFFChannelUnit] = DCFFChannelUnit.init_from_graph(graph)
        mutable_units = [
            DCFFChannelUnit.init_from_channel_unit(unit) for unit in units
        ]
        model = model.to(DEVICE)
        self._test_units(mutable_units, model)

    def _test_units(self, units: List[DCFFChannelUnit], model):
        for unit in units:
            unit.prepare_for_pruning(model)
        mutable_units = [unit for unit in units if unit.is_mutable]
        self.assertGreaterEqual(len(mutable_units), 1)
        for unit in mutable_units:
            choice = unit.sample_choice()
            unit.current_choice = choice
            for channel in unit.output_related:
                if isinstance(channel.module, FuseConv2d):
                    layeri_softmaxp = channel.module.get_pooled_channel(1.0)
                    # update fuseconv op's selected layeri_softmax
                    channel.module.set_forward_args(choice=layeri_softmaxp)
        x = torch.rand([2, 3, 224, 224]).to(DEVICE)
        y = model(x)
        self.assertSequenceEqual(y.shape, [2, 1000])
