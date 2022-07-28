# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
from unittest import TestCase

import pytest
import torch
from torch import nn

from mmrazor.models.architectures import (CenterCropDynamicConv2d,
                                          DynamicBatchNorm1d,
                                          DynamicBatchNorm2d,
                                          DynamicBatchNorm3d, DynamicConv2d,
                                          DynamicLinear, DynamicSequential,
                                          ProgressiveDynamicConv2d)
from mmrazor.models.architectures.dynamic_op import DynamicOP
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.models.subnet import export_fix_mutable, load_fix_subnet


class TestDynamicOP(TestCase):

    def test_dynamic_conv2d(self) -> None:
        d_conv2d = DynamicConv2d(
            in_channels=4, out_channels=10, kernel_size=3, stride=1, bias=True)

        x_max = torch.rand(10, 4, 224, 224)
        out_before_mutate = d_conv2d(x_max)

        in_channels_mutable = OneShotMutableChannel(
            4, candidate_choices=[2, 3, 4], candidate_mode='number')
        out_channels_mutable = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')
        d_conv2d.mutate_in_channels(in_channels_mutable)
        d_conv2d.mutate_out_channels(out_channels_mutable)

        with pytest.raises(RuntimeError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 4
        d_conv2d.mutate_out_channels = 10

        out_max = d_conv2d(x_max)
        assert torch.equal(out_before_mutate, out_max)

        d_conv2d.mutable_in.current_choice = 3
        d_conv2d.mutable_out.current_choice = 4

        x = torch.rand(10, 3, 224, 224)
        out1 = d_conv2d(x)
        assert out1.size(1) == 4

        fix_mutables = export_fix_mutable(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)

        s_conv2d = d_conv2d.to_static_op()
        assert s_conv2d.weight.size(0) == 4
        assert s_conv2d.weight.size(1) == 3
        assert s_conv2d.bias.size(0) == 4
        out2 = s_conv2d(x)

        assert torch.equal(out1, out2)

    def test_dynamic_conv2d_single_mutable(self) -> None:
        d_conv2d = DynamicConv2d(
            in_channels=4, out_channels=10, kernel_size=3, stride=1, bias=True)
        in_channels_mutable = OneShotMutableChannel(
            4, candidate_choices=[2, 3, 4], candidate_mode='number')

        d_conv2d.mutate_in_channels(in_channels_mutable)

        with pytest.raises(RuntimeError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 3
        assert d_conv2d.mutable_out is None

        x = torch.rand(10, 3, 224, 224)
        out1 = d_conv2d(x)
        assert out1.size(1) == 10

        with pytest.raises(RuntimeError):
            _ = d_conv2d.to_static_op()

        fix_mutables = export_fix_mutable(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)

        s_conv2d = d_conv2d.to_static_op()
        assert s_conv2d.weight.size(0) == 10
        assert s_conv2d.weight.size(1) == 3
        out2 = s_conv2d(x)

        assert torch.equal(out1, out2)

    def test_dynamic_conv2d_depthwise(self) -> None:
        d_conv2d = DynamicConv2d(
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)

        in_channels_mutable = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')
        out_channels_mutable = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')

        d_conv2d.mutate_in_channels(in_channels_mutable)
        d_conv2d.mutate_out_channels(out_channels_mutable)

        with pytest.raises(RuntimeError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 8
        d_conv2d.mutable_out.current_choice = 8

        x = torch.rand(10, 8, 224, 224)
        out1 = d_conv2d(x)
        assert out1.size(1) == 8

        with pytest.raises(RuntimeError):
            _ = d_conv2d.to_static_op()

        fix_mutables = export_fix_mutable(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)

        s_conv2d = d_conv2d.to_static_op()
        assert s_conv2d.weight.size(0) == 8
        assert s_conv2d.weight.size(1) == 1
        assert s_conv2d.bias.size(0) == 8
        out2 = s_conv2d(x)

        assert torch.equal(out1, out2)

    def test_dynamic_linear(self) -> None:
        in_features_mutable = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')
        out_features_mutable = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')

        d_linear = DynamicLinear(in_features=10, out_features=10, bias=True)
        d_linear.mutate_in_features(in_features_mutable)
        d_linear.mutate_out_features(out_features_mutable)

        with pytest.raises(RuntimeError):
            d_linear.to_static_op()

        d_linear.mutable_in.current_choice = 8
        d_linear.mutable_out.current_choice = 4

        x = torch.rand(10, 8)
        out1 = d_linear(x)
        self.assertEqual(out1.size(1), 4)

        with pytest.raises(RuntimeError):
            _ = d_linear.to_static_op()

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

    def test_dynamic_sequential(self) -> None:
        depth_mutable = OneShotMutableValue(
            value_list=[2, 3, 4, 5], default_value=5)

        modules = [
            nn.Conv2d(3, 32, 3),
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 256, 3),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 64, 3),
            nn.Conv2d(64, 32, 3)
        ]

        with pytest.raises(ValueError):
            d_seq = DynamicSequential(*modules)
            d_seq.mutate_depth(depth_mutable)

        modules = modules[:-1]
        d_seq = DynamicSequential(*modules)
        d_seq.mutate_depth(depth_mutable)

        d_seq.depth_mutable.current_choice = 3

        x = torch.rand(10, 3, 224, 224)
        out1 = d_seq(x)
        out2 = nn.Sequential(*modules[:3])(x)
        assert torch.equal(out1, out2)

        with pytest.raises(RuntimeError):
            _ = d_seq.to_static_op()
        d_seq.depth_mutable.fix_chosen(d_seq.depth_mutable.dump_chosen())
        s_seq = d_seq.to_static_op()
        assert isinstance(s_seq, nn.Sequential)
        assert not isinstance(s_seq, DynamicSequential)

        out3 = s_seq(x)
        assert torch.equal(out3, out2)


# TODO
# unittest not support parametrize
@pytest.mark.parametrize('dynamic_class,input_shape',
                         [(DynamicBatchNorm1d, (10, 8, 224)),
                          (DynamicBatchNorm2d, (10, 8, 224, 224)),
                          (DynamicBatchNorm3d, (10, 8, 3, 224, 224))])
def test_dynamic_bn(dynamic_class: nn.Module, input_shape: Tuple[int]) -> None:
    num_features_mutable = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')

    d_bn = dynamic_class(num_features=10)
    d_bn.mutate_num_features(num_features_mutable)

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

    in_channels_mutable = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    out_channels_mutable = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    kernel_size_mutable = OneShotMutableValue(value_list=[3, 5, 7])

    with pytest.raises(ValueError):
        d_conv2d = dynamic_class(
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)
        d_conv2d.mutate_kernel_size(kernel_size_mutable)

    d_conv2d = dynamic_class(
        in_channels=10,
        out_channels=10,
        groups=1,
        kernel_size=7,
        stride=1,
        bias=True)
    d_conv2d.mutate_in_channels(in_channels_mutable)
    d_conv2d.mutate_out_channels(out_channels_mutable)
    d_conv2d.mutate_kernel_size(kernel_size_mutable)

    with pytest.raises(RuntimeError):
        d_conv2d.to_static_op()

    d_conv2d.mutable_in.current_choice = 8
    d_conv2d.mutable_out.current_choice = 8
    if mutate_kernel_size:
        d_conv2d.mutable_kernel_size.current_choice = 5

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
    if mutate_kernel_size:
        assert s_conv2d.kernel_size == (5, 5)
        assert tuple(s_conv2d.weight.shape[2:]) == (5, 5)
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)