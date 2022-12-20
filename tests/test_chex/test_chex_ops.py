# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmrazor.models.chex import ChexConv2d
from mmrazor.models.mutables import SimpleMutableChannel


class TestChexOps(unittest.TestCase):

    def test_ops(self):
        for in_c, out_c in [(4, 8), (8, 4), (8, 8)]:
            conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
            conv: ChexConv2d = ChexConv2d.convert_from(conv)

            mutable_in = SimpleMutableChannel(in_c)
            mutable_out = SimpleMutableChannel(out_c)

            conv.register_mutable_attr('in_channels', mutable_in)
            conv.register_mutable_attr('out_channels', mutable_out)

            mutable_out.current_choice = torch.normal(0, 1, [out_c]) < 0

            self.assertEqual(list(conv.prune_imp(4).shape), [out_c])
            self.assertEqual(list(conv.growth_imp.shape), [out_c])
