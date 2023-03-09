# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor import digit_version
from mmrazor.models.mutables import SimpleMutableChannel
from mmrazor.models.utils.expandable_utils import (
    expand_expandable_dynamic_model, make_channel_divisible,
    to_expandable_model)
from mmrazor.models.utils.expandable_utils.ops import ExpandLinear
from ....data.models import DwConvModel, MultiConcatModel, SingleLineModel


class TestExpand(unittest.TestCase):

    def check_torch_version(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')

    def test_expand(self):
        self.check_torch_version()
        for Model in [MultiConcatModel, DwConvModel]:
            x = torch.rand([1, 3, 224, 224])
            model = Model()
            print(model)
            mutator = to_expandable_model(model)
            print(mutator.choice_template)
            print(model)
            y1 = model(x)

            for unit in mutator.mutable_units:
                unit.expand(10)
                print(unit.mutable_channel.mask.shape)
            expand_expandable_dynamic_model(model, zero=True)
            print(model)
            y2 = model(x)
            self.assertTrue((y1 - y2).abs().max() < 1e-3)

    def test_expand_static_model(self):
        self.check_torch_version()
        x = torch.rand([1, 3, 224, 224])
        model = SingleLineModel()
        y1 = model(x)
        make_channel_divisible(model, divisor=4)
        y2 = model(x)
        print(y1.reshape([-1])[:5])
        print(y2.reshape([-1])[:5])
        self.assertTrue((y1 - y2).abs().max() < 1e-3)

    def test_ExpandConv2d(self):
        self.check_torch_version()
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
