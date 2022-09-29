# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import SequentialMutableChannelUnit


class TestSequentialMutableChannelUnit(TestCase):

    def test_num(self):
        group = SequentialMutableChannelUnit(48)
        group.current_choice = 24
        self.assertEqual(group.current_choice, 24)

        group.current_choice = 0.5
        self.assertEqual(group.current_choice, 24)

    def test_ratio(self):
        group = SequentialMutableChannelUnit(48, choice_mode='ratio')
        group.current_choice = 0.5
        self.assertEqual(group.current_choice, 0.5)
        group.current_choice = 24
        self.assertEqual(group.current_choice, 0.5)

    def test_divisor(self):
        group = SequentialMutableChannelUnit(
            48, choice_mode='number', divisor=8)
        group.current_choice = 20
        self.assertEqual(group.current_choice, 24)
        self.assertTrue(group.sample_choice() % 8 == 0)

        group = SequentialMutableChannelUnit(
            48, choice_mode='ratio', divisor=8)
        group.current_choice = 0.3
        self.assertEqual(group.current_choice, 1 / 3)

    def test_config_template(self):
        group = SequentialMutableChannelUnit(
            48, choice_mode='ratio', divisor=8)
        config = group.config_template(with_init_args=True)
        group2 = SequentialMutableChannelUnit.init_from_cfg(None, config)
        self.assertDictEqual(
            group2.config_template(with_init_args=True)['init_args'],
            config['init_args'])
