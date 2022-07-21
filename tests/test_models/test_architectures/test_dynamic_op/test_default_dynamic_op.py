# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
from unittest import TestCase

import pytest
import torch
from torch import nn

from mmrazor.models.architectures import (
    CenterCropDynamicConv2d, DynamicBatchNorm1d, DynamicBatchNorm2d,
    DynamicBatchNorm3d, DynamicConv2d, DynamicLinear, ProgressiveDynamicConv2d)
from mmrazor.models.architectures.dynamic_op import DynamicOP
from mmrazor.models.subnet import export_fix_mutable, load_fix_subnet


class TestDefaultDynamicOP(TestCase):

    def test_dynamic_conv2d(self) -> None:
        in_channels_cfg = dict(type='OneShotMutableChannel')
        out_channels_cfg = dict(type='OneShotMutableChannel')
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

        with pytest.raises(RuntimeError):
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
        assert s_conv2d.weight.size(0) == 4
        assert s_conv2d.weight.size(1) == 3
        assert s_conv2d.bias.size(0) == 4
        out2 = s_conv2d(x)

        self.assertTrue(torch.equal(out1, out2))

    def test_dynamic_conv2d_depthwise(self) -> None:
        in_channels_cfg = dict(
            type='OneShotMutableChannel',
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        out_channels_cfg = dict(
            type='OneShotMutableChannel',
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

        with pytest.raises(RuntimeError):
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
        assert s_conv2d.weight.size(0) == 8
        assert s_conv2d.weight.size(1) == 1
        assert s_conv2d.bias.size(0) == 8
        out2 = s_conv2d(x)
>>>>>>> remove slimmable channel mutable and refactor dynamic op

        self.assertTrue(torch.equal(out1, out2))

    def test_dynamic_linear(self) -> None:
        in_features_cfg = dict(
            type='OneShotMutableChannel',
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        out_features_cfg = dict(
            type='OneShotMutableChannel',
            candidate_mode='number',
            candidate_choices=[4, 8, 10])
        mutable_cfgs = dict(
            in_features=in_features_cfg, out_features=out_features_cfg)

        d_linear = DynamicLinear(
            mutable_cfgs=mutable_cfgs,
            in_features=10,
            out_features=10,
            bias=True)

        with pytest.raises(RuntimeError):
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
        assert s_linear.weight.size(0) == 4
        assert s_linear.weight.size(1) == 8
        assert s_linear.bias.size(0) == 4
        assert not isinstance(s_linear, DynamicOP)
        assert isinstance(s_linear, nn.Linear)
        out2 = s_linear(x)

        self.assertTrue(torch.equal(out1, out2))


# TODO
# unittest not support parametrize
@pytest.mark.parametrize('dynamic_class,input_shape',
                         [(DynamicBatchNorm1d, (10, 8, 224)),
                          (DynamicBatchNorm2d, (10, 8, 224, 224)),
                          (DynamicBatchNorm3d, (10, 8, 3, 224, 224))])
def test_dynamic_bn(dynamic_class: nn.Module, input_shape: Tuple[int]) -> None:
    num_features_cfg = dict(
        type='OneShotMutableChannel',
        candidate_mode='number',
        candidate_choices=[4, 8, 10])
    mutable_cfgs = dict(num_features=num_features_cfg)

    d_bn = dynamic_class(mutable_cfgs=mutable_cfgs, num_features=10)

    with pytest.raises(RuntimeError):
        d_bn.to_static_op()

    d_bn.mutable_in.current_choice = 8

    x = torch.rand(*input_shape)
    out1 = d_bn(x)
    assert out1.size(1) == 8

    fix_mutables = export_fix_mutable(d_bn)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_bn, fix_mutables)
    assert isinstance(d_bn, dynamic_class)
    assert isinstance(d_bn, DynamicOP)

    s_bn = d_bn.to_static_op()
    assert s_bn.weight.size(0) == 8
    assert s_bn.bias.size(0) == 8
    assert s_bn.running_mean.size(0) == 8
    assert s_bn.running_var.size(0) == 8
    assert not isinstance(s_bn, DynamicOP)
    assert isinstance(s_bn, getattr(nn, d_bn.batch_norm_type))
    out2 = s_bn(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize('dynamic_class',
                         [ProgressiveDynamicConv2d, CenterCropDynamicConv2d])
def test_kernel_dynamic_conv2d(dynamic_class: nn.Module) -> None:
    in_channels_cfg = dict(
        type='OneShotMutableChannel',
        candidate_mode='number',
        candidate_choices=[4, 8, 10])
    out_channels_cfg = dict(
        type='OneShotMutableChannel',
        candidate_mode='number',
        candidate_choices=[4, 8, 10])
    kernel_size_cfg = dict(type='OneShotMutableValue', value_list=[3, 5, 7])
    mutable_cfgs = dict(
        in_channels=in_channels_cfg,
        out_channels=out_channels_cfg,
        kernel_size=kernel_size_cfg)

    with pytest.raises(ValueError):
        d_conv2d = dynamic_class(
            mutable_cfgs=mutable_cfgs,
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)

    d_conv2d = dynamic_class(
        mutable_cfgs=mutable_cfgs,
        in_channels=10,
        out_channels=10,
        groups=1,
        kernel_size=7,
        stride=1,
        bias=True)

    with pytest.raises(RuntimeError):
        d_conv2d.to_static_op()

    d_conv2d.mutable_in.current_choice = 8
    d_conv2d.mutable_out.current_choice = 8
    d_conv2d.kernel_size_mutable.current_choice = 5

    x = torch.rand(10, 8, 224, 224)
    out1 = d_conv2d(x)
    assert out1.size(1) == 8

    fix_mutables = export_fix_mutable(d_conv2d)
    print(fix_mutables)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_conv2d, fix_mutables)

    s_conv2d = d_conv2d.to_static_op()
    assert s_conv2d.weight.size(0) == 8
    assert s_conv2d.weight.size(1) == 8
    assert s_conv2d.bias.size(0) == 8
    assert s_conv2d.kernel_size == (5, 5)
    assert tuple(s_conv2d.weight.shape[2:]) == (5, 5)
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)
