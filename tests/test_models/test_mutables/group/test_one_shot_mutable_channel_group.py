# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import OneShotMutableChannelUnit


class TestSequentialMutableChannelUnit(TestCase):

    def test_init(self):
        group = OneShotMutableChannelUnit(
            48, [20, 30, 40], candidate_mode='number', divisor=8)
        self.assertSequenceEqual(group.candidate_choices, [24, 32, 40])

        group = OneShotMutableChannelUnit(
            48, [0.3, 0.5, 0.7], candidate_mode='ratio', divisor=8)
        self.assertSequenceEqual(group.candidate_choices, [1 / 3, 0.5, 2 / 3])
