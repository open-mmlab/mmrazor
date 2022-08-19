# Copyright (c) OpenMMLab. All rights reserved.
"""from unittest import TestCase.

import pytest
import torch

from mmrazor.models.architectures import DynamicConv2d
from mmrazor.structures import export_fix_subnet, load_fix_subnet
from .utils import fix_dynamic_op

class TestDefaultDynamicOP(TestCase):

    def test_dynamic_conv2d(self) -> None:
        in_channels_cfg = dict(type='SlimmableMutableChannel', num_channels=4)
        out_channels_cfg = dict(
            type='SlimmableMutableChannel', num_channels=10)

        d_conv2d = DynamicConv2d(
            in_channels_cfg,
            out_channels_cfg,
            in_channels=4,
            out_channels=10,
            kernel_size=3,
            stride=1,
            bias=True)

        d_conv2d.mutable_in.candidate_choices = [2, 3, 4]
        d_conv2d.mutable_out.candidate_choices = [4, 8, 10]

        with pytest.raises(AssertionError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 1
        d_conv2d.mutable_out.current_choice = 0

        x = torch.rand(10, 3, 224, 224)
        out1 = d_conv2d(x)
        self.assertEqual(out1.size(1), 4)

        fix_mutables = export_fix_subnet(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)
        fix_dynamic_op(d_conv2d, fix_mutables)

        out2 = d_conv2d(x)
        self.assertTrue(torch.equal(out1, out2))

        s_conv2d = d_conv2d.to_static_op()
        out3 = s_conv2d(x)

        self.assertTrue(torch.equal(out1, out3))

    def test_dynamic_conv2d_depthwise(self) -> None:
        in_channels_cfg = dict(type='SlimmableMutableChannel', num_channels=10)
        out_channels_cfg = dict(
            type='SlimmableMutableChannel', num_channels=10)

        d_conv2d = DynamicConv2d(
            in_channels_cfg,
            out_channels_cfg,
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)

        d_conv2d.mutable_in.candidate_choices = [4, 8, 10]
        d_conv2d.mutable_out.candidate_choices = [4, 8, 10]

        with pytest.raises(AssertionError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 1
        d_conv2d.mutable_out.current_choice = 1

        x = torch.rand(10, 8, 224, 224)
        out1 = d_conv2d(x)
        self.assertEqual(out1.size(1), 8)

        fix_mutables = export_fix_subnet(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)
        fix_dynamic_op(d_conv2d, fix_mutables)

        out2 = d_conv2d(x)
        self.assertTrue(torch.equal(out1, out2))

        s_conv2d = d_conv2d.to_static_op()
        out3 = s_conv2d(x)

        self.assertTrue(torch.equal(out1, out3))
"""
