# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch.nn as nn

from mmrazor.models.mutables import L1MutableChannelGroup
from mmrazor.models.mutators import BaseChannelMutator
from ....data.models import LineModel


class TestL1MutableChannelGroup(TestCase):

    def test_init(self):
        model = LineModel()
        mutator = BaseChannelMutator(
            channel_group_cfg={
                'type': 'L1MutableChannelGroup',
                'default_args': {
                    'choice_mode': 'ratio'
                }
            })
        mutator.prepare_from_supernet(model)
        mutator.set_choices(mutator.sample_choices())
        print(mutator.groups)
        print(mutator.mutable_groups)
        print(mutator.choice_template)

    def test_convnd(self):
        group = L1MutableChannelGroup(8)
        conv = nn.Conv3d(3, 8, 3)
        norm = group._get_l1_norm(conv, 0, 8)
        self.assertSequenceEqual(norm.shape, [8])
