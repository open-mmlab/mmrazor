# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Type
from unittest import TestCase

import pytest
import torch
from torch import nn

from mmrazor.models.architectures.dynamic_op.bricks import (
    CenterCropDynamicConv2d, DynamicConv2d, ProgressiveDynamicConv2d)
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet
from ..utils import fix_dynamic_op


class TestDynamicConv2d(TestCase):

    def test_dynamic_conv2d_depthwise(self) -> None:
        d_conv2d = DynamicConv2d(
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)

        mutable_in_channels = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')
        mutable_out_channels = OneShotMutableChannel(
            10, candidate_choices=[4, 8, 10], candidate_mode='number')

        d_conv2d.mutate_in_channels(mutable_in_channels)
        d_conv2d.mutate_out_channels(mutable_out_channels)

        with pytest.raises(RuntimeError):
            d_conv2d.to_static_op()

        d_conv2d.mutable_in.current_choice = 8
        d_conv2d.mutable_out.current_choice = 8

        x = torch.rand(10, 8, 224, 224)
        out1 = d_conv2d(x)
        assert out1.size(1) == 8

        with pytest.raises(RuntimeError):
            _ = d_conv2d.to_static_op()

        fix_mutables = export_fix_subnet(d_conv2d)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_conv2d, fix_mutables)
        fix_dynamic_op(d_conv2d, fix_mutables)

        s_conv2d = d_conv2d.to_static_op()
        assert s_conv2d.weight.size(0) == 8
        assert s_conv2d.weight.size(1) == 1
        assert s_conv2d.bias.size(0) == 8
        out2 = s_conv2d(x)

        assert torch.equal(out1, out2)


@pytest.mark.parametrize('bias', [True, False])
def test_dynamic_conv2d(bias: bool) -> None:
    d_conv2d = DynamicConv2d(
        in_channels=4, out_channels=10, kernel_size=3, stride=1, bias=bias)

    x_max = torch.rand(10, 4, 224, 224)
    out_before_mutate = d_conv2d(x_max)

    mutable_in_channels = OneShotMutableChannel(
        4, candidate_choices=[2, 3, 4], candidate_mode='number')
    mutable_out_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    d_conv2d.mutate_in_channels(mutable_in_channels)
    d_conv2d.mutate_out_channels(mutable_out_channels)

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

    fix_mutables = export_fix_subnet(d_conv2d)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_conv2d, fix_mutables)
    fix_dynamic_op(d_conv2d, fix_mutables)

    s_conv2d = d_conv2d.to_static_op()
    assert s_conv2d.weight.size(0) == 4
    assert s_conv2d.weight.size(1) == 3
    if bias:
        assert s_conv2d.bias.size(0) == 4
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize(
    ['is_mutate_in_channels', 'in_channels', 'out_channels'], [(True, 6, 10),
                                                               (False, 10, 4)])
def test_dynamic_conv2d_mutable_single_channels(is_mutate_in_channels: bool,
                                                in_channels: int,
                                                out_channels: int) -> None:
    d_conv2d = DynamicConv2d(
        in_channels=10, out_channels=10, kernel_size=3, stride=1, bias=True)
    mutable_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 6, 10], candidate_mode='number')

    if is_mutate_in_channels:
        d_conv2d.mutate_in_channels(mutable_channels)
    else:
        d_conv2d.mutate_out_channels(mutable_channels)

    with pytest.raises(RuntimeError):
        d_conv2d.to_static_op()

    if is_mutate_in_channels:
        d_conv2d.mutable_in.current_choice = in_channels
        assert d_conv2d.mutable_out is None
    else:
        d_conv2d.mutable_out_channels.current_choice = out_channels
        assert d_conv2d.mutable_in is None

    x = torch.rand(3, in_channels, 224, 224)
    out1 = d_conv2d(x)

    assert out1.size(1) == out_channels

    with pytest.raises(RuntimeError):
        _ = d_conv2d.to_static_op()

    fix_mutables = export_fix_subnet(d_conv2d)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_conv2d, fix_mutables)
    fix_dynamic_op(d_conv2d, fix_mutables)

    s_conv2d = d_conv2d.to_static_op()
    assert s_conv2d.weight.size(0) == out_channels
    assert s_conv2d.weight.size(1) == in_channels
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize('dynamic_class',
                         [ProgressiveDynamicConv2d, CenterCropDynamicConv2d])
@pytest.mark.parametrize('kernel_size_list', [None, [5], [3, 5, 7]])
def test_kernel_dynamic_conv2d(dynamic_class: Type[nn.Conv2d],
                               kernel_size_list: bool) -> None:

    mutable_in_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    mutable_out_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    mutable_kernel_size = None
    if kernel_size_list is not None:
        mutable_kernel_size = OneShotMutableValue(value_list=kernel_size_list)

    d_conv2d = dynamic_class(
        in_channels=10,
        out_channels=10,
        groups=1,
        kernel_size=3 if kernel_size_list is None else max(kernel_size_list),
        stride=1,
        bias=True)
    d_conv2d.mutate_in_channels(mutable_in_channels)
    d_conv2d.mutate_out_channels(mutable_out_channels)
    if kernel_size_list is not None:
        copied_mutable_kernel_size = copy.deepcopy(mutable_kernel_size)
        copied_d_conv2d = copy.deepcopy(d_conv2d)

        copied_mutable_kernel_size._value_list = []
        with pytest.raises(ValueError):
            _ = copied_d_conv2d.mutate_kernel_size(copied_mutable_kernel_size)

        wrong_kernel_size_list = [4]
        with pytest.raises(ValueError):
            copied_d_conv2d.mutate_kernel_size(copied_mutable_kernel_size,
                                               wrong_kernel_size_list)

        copied_d_conv2d.mutate_kernel_size(copied_mutable_kernel_size,
                                           kernel_size_list)
        d_conv2d.mutate_kernel_size(mutable_kernel_size)
        assert d_conv2d.kernel_size_list == copied_d_conv2d.kernel_size_list

    with pytest.raises(RuntimeError):
        d_conv2d.to_static_op()

    d_conv2d.mutable_in.current_choice = 8
    d_conv2d.mutable_out.current_choice = 8
    if kernel_size_list is not None:
        kernel_size = mutable_kernel_size.sample_choice()
        d_conv2d.mutable_kernel_size.current_choice = kernel_size

    x = torch.rand(3, 8, 224, 224)
    out1 = d_conv2d(x)
    assert out1.size(1) == 8

    fix_mutables = export_fix_subnet(d_conv2d)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_conv2d, fix_mutables)
    fix_dynamic_op(d_conv2d, fix_mutables)

    s_conv2d = d_conv2d.to_static_op()
    assert s_conv2d.weight.size(0) == 8
    assert s_conv2d.weight.size(1) == 8
    assert s_conv2d.bias.size(0) == 8
    if kernel_size_list is not None:
        assert s_conv2d.kernel_size == (kernel_size, kernel_size)
        assert tuple(s_conv2d.weight.shape[2:]) == (kernel_size, kernel_size)
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize('dynamic_class',
                         [ProgressiveDynamicConv2d, CenterCropDynamicConv2d])
def test_mutable_kernel_dynamic_conv2d_grad(
        dynamic_class: Type[nn.Conv2d]) -> None:
    from mmrazor.models.architectures.dynamic_op.bricks.dynamic_conv import \
        _get_current_kernel_pos

    kernel_size_list = [3, 5, 7]
    d_conv2d = dynamic_class(
        in_channels=3,
        out_channels=10,
        groups=1,
        kernel_size=max(kernel_size_list),
        stride=1,
        bias=False)

    mutable_kernel_size = OneShotMutableValue(value_list=kernel_size_list)
    d_conv2d.mutate_kernel_size(mutable_kernel_size)

    x = torch.rand(3, 3, 224, 224, requires_grad=True)

    for kernel_size in kernel_size_list:
        mutable_kernel_size.current_choice = kernel_size
        out = d_conv2d(x).sum()
        out.backward()

        start_offset, end_offset = _get_current_kernel_pos(
            max(kernel_size_list), kernel_size)

        mask = torch.ones_like(
            d_conv2d.weight, requires_grad=False, dtype=torch.bool)
        mask[:, :, start_offset:end_offset, start_offset:end_offset] = 0
        assert d_conv2d.weight.grad[mask].norm().item() == 0

        d_conv2d.weight.grad.zero_()
