# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import SequentialMutableChannelUnit


class TestSequentialMutableChannelUnit(TestCase):

    def test_num(self):
        unit = SequentialMutableChannelUnit(48)
        unit.current_choice = 24
        self.assertEqual(unit.current_choice, 24)

        unit.current_choice = 0.5
        self.assertEqual(unit.current_choice, 24)

    def test_ratio(self):
        unit = SequentialMutableChannelUnit(48, choice_mode='ratio')
        unit.current_choice = 0.5
        self.assertEqual(unit.current_choice, 0.5)
        unit.current_choice = 24
        self.assertEqual(unit.current_choice, 0.5)

    def test_divisor(self):
        unit = SequentialMutableChannelUnit(
            48, choice_mode='number', divisor=8)
        unit.current_choice = 20
        self.assertEqual(unit.current_choice, 24)
        self.assertTrue(unit.sample_choice() % 8 == 0)

        unit = SequentialMutableChannelUnit(48, choice_mode='ratio', divisor=8)
        unit.current_choice = 0.3
        self.assertEqual(unit.current_choice, 1 / 3)

    def test_config_template(self):
        unit = SequentialMutableChannelUnit(48, choice_mode='ratio', divisor=8)
        config = unit.config_template(with_init_args=True)
        unit2 = SequentialMutableChannelUnit.init_from_cfg(None, config)
        self.assertDictEqual(
            unit2.config_template(with_init_args=True)['init_args'],
            config['init_args'])
