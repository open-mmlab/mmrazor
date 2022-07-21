# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from torch import nn

from mmrazor.models.architectures import DynamicConv2d, DynamicLinear
from mmrazor.models.architectures.dynamic_op import DynamicOP
from mmrazor.models.subnet import export_fix_mutable, load_fix_subnet


class TestDefaultDynamicOP(TestCase):

    def test_dynamic_conv2d(self) -> None:
        in_channels_cfg = dict(type='OneShotMutableChannel', num_channels=4)
        out_channels_cfg = dict(type='OneShotMutableChannel', num_channels=10)
        mutable_cfgs = dict(
            in_channels=in_channels_cfg, out_channels=out_channels_cfg)

        d_conv2d = DynamicConv2d(
            mutable_cfgs=mutable_cfgs,
            in_channels=4,
            out_channels=10,
            kernel_size=3,
            stride=1,
            bias=True)

        d_conv2d.mutable_in.set_candidate_choices('number', [2, 3, 4])
        d_conv2d.mutable_out.set_candidate_choices('number', [4, 8, 10])

        with pytest.raises(AssertionError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 3
        d_conv2d.mutable_out.current_choice = 4

        x = torch.rand(10, 3, 224, 224)
        out1 = d_conv2d(x)
        self.assertEqual(out1.size(1), 4)

<<<<<<< HEAD
        fix_mutables = export_fix_subnet(d_conv2d)
        load_fix_subnet(d_conv2d, fix_mutables)

        out2 = d_conv2d(x)
        self.assertTrue(torch.equal(out1, out2))
=======
        fix_mutables = export_fix_mutable(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)
>>>>>>> remove slimmable channel mutable and refactor dynamic op

        s_conv2d = d_conv2d.to_static_op()
        out2 = s_conv2d(x)

        self.assertTrue(torch.equal(out1, out2))

    def test_dynamic_conv2d_depthwise(self) -> None:
        in_channels_cfg = dict(
            type='OneShotMutableChannel',
            num_channels=4,
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        out_channels_cfg = dict(
            type='OneShotMutableChannel',
            num_channels=10,
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        mutable_cfgs = dict(
            in_channels=in_channels_cfg, out_channels=out_channels_cfg)

        d_conv2d = DynamicConv2d(
            mutable_cfgs=mutable_cfgs,
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)

        with pytest.raises(AssertionError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 8
        d_conv2d.mutable_out.current_choice = 8

        x = torch.rand(10, 8, 224, 224)
        out1 = d_conv2d(x)
        self.assertEqual(out1.size(1), 8)

<<<<<<< HEAD
        fix_mutables = export_fix_subnet(d_conv2d)
        load_fix_subnet(d_conv2d, fix_mutables)
=======
        fix_mutables = export_fix_mutable(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)

        s_conv2d = d_conv2d.to_static_op()
        out2 = s_conv2d(x)
>>>>>>> remove slimmable channel mutable and refactor dynamic op

        self.assertTrue(torch.equal(out1, out2))

    def test_dynamic_linear(self) -> None:
        in_features_cfg = dict(
            type='OneShotMutableChannel',
            num_channels=4,
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        out_features_cfg = dict(
            type='OneShotMutableChannel',
            num_channels=10,
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        mutable_cfgs = dict(
            in_features=in_features_cfg, out_features=out_features_cfg)

        d_linear = DynamicLinear(
            mutable_cfgs=mutable_cfgs,
            in_features=10,
            out_features=10,
            bias=True)

        with pytest.raises(AssertionError):
            d_linear.to_static_op()

        d_linear.mutable_in.current_choice = 8
        d_linear.mutable_out.current_choice = 4

        x = torch.rand(10, 8)
        out1 = d_linear(x)
        self.assertEqual(out1.size(1), 4)

        fix_mutables = export_fix_mutable(d_linear)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_linear, fix_mutables)
        assert isinstance(d_linear, nn.Linear)
        assert isinstance(d_linear, DynamicOP)

        s_linear = d_linear.to_static_op()
        assert not isinstance(s_linear, DynamicOP)
        assert isinstance(s_linear, nn.Linear)
        out2 = s_linear(x)

        self.assertTrue(torch.equal(out1, out2))
