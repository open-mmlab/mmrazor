# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch.nn as nn

from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.models.mutators import ChannelMutator
from .....data.models import SingleLineModel


class TestL1MutableChannelUnit(TestCase):

    def test_init(self):
        model = SingleLineModel()
        mutator = ChannelMutator(
            channel_unit_cfg={
                'type': 'L1MutableChannelUnit',
                'default_args': {
                    'choice_mode': 'ratio'
                }
            })
        mutator.prepare_from_supernet(model)

    def test_convnd(self):
        unit = L1MutableChannelUnit(8)
        conv = nn.Conv3d(3, 8, 3)
        norm = unit._get_l1_norm(conv, 0, 8)
        self.assertSequenceEqual(norm.shape, [8])
