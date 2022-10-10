# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import OneShotMutableChannelUnit


class TestSequentialMutableChannelUnit(TestCase):

    def test_init(self):
        unit = OneShotMutableChannelUnit(
            48, [20, 30, 40], choice_mode='number', divisor=8)
        self.assertSequenceEqual(unit.candidate_choices, [24, 32, 40])

        unit = OneShotMutableChannelUnit(
            48, [0.3, 0.5, 0.7], choice_mode='ratio', divisor=8)
        self.assertSequenceEqual(unit.candidate_choices, [1 / 3, 0.5, 2 / 3])
