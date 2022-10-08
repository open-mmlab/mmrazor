# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models.mutables import DCFFChannelUnit


class TestDCFFChannelUnit(TestCase):

    def test_init(self):
        unit = DCFFChannelUnit(48, [20], candidate_mode='number')
        self.assertSequenceEqual(unit.candidate_choices, [20])

        unit = DCFFChannelUnit(48, [0.5], candidate_mode='ratio')
        self.assertSequenceEqual(unit.candidate_choices, [0.5])
