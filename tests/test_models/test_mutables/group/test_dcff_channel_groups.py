# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from unittest import TestCase

from mmrazor.models.mutables import DCFFChannelGroup
from mmrazor.registry import MODELS


class TestDCFFChannelGroup(TestCase):

    def test_init(self):
        group = DCFFChannelGroup(
            48, [20], candidate_mode='number')
        self.assertSequenceEqual(group.candidate_choices, [20])

        group = DCFFChannelGroup(
            48, [0.5], candidate_mode='ratio')
        self.assertSequenceEqual(group.candidate_choices, [0.5])
