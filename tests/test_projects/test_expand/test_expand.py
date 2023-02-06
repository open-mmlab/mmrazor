# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutables import SimpleMutableChannel
from mmrazor.models.mutators import ChannelMutator
from projects.cores.expandable_ops.ops import ExpandLinear
from projects.cores.expandable_ops.unit import (ExpandableUnit, expand_model,
                                                expand_static_model)
from ...data.models import MultiConcatModel, SingleLineModel


class TestExpand(unittest.TestCase):

    def test_expand(self):
        x = torch.rand([1, 3, 224, 224])
        model = MultiConcatModel()
        print(model)
        mutator = ChannelMutator[ExpandableUnit](
            channel_unit_cfg=ExpandableUnit)
        mutator.prepare_from_supernet(model)
        print(mutator.choice_template)
        print(model)
        y1 = model(x)

        for unit in mutator.mutable_units:
            unit.expand(10)
            print(unit.mutable_channel.mask.shape)
        expand_model(model, zero=True)
        print(model)
        y2 = model(x)
        self.assertTrue((y1 - y2).abs().max() < 1e-3)

    def test_expand_static_model(self):
        x = torch.rand([1, 3, 224, 224])
        model = SingleLineModel()
        y1 = model(x)
        expand_static_model(model, divisor=4)
        y2 = model(x)
        print(y1.reshape([-1])[:5])
        print(y2.reshape([-1])[:5])
        self.assertTrue((y1 - y2).abs().max() < 1e-3)

    def test_ExpandConv2d(self):
        linear = ExpandLinear(3, 3)
        mutable_in = SimpleMutableChannel(3)
        mutable_out = SimpleMutableChannel(3)
        linear.register_mutable_attr('in_channels', mutable_in)
        linear.register_mutable_attr('out_channels', mutable_out)

        print(linear.weight)

        mutable_in.mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        mutable_out.mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        linear_ex = linear.expand(zero=True)
        print(linear_ex.weight)
