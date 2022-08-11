# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import nn

from mmrazor.models.architectures.dynamic_op.bricks import (
    CenterCropDynamicConv2d, ProgressiveDynamicConv2d)
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet
from ..utils import _fix_dynamic_op


@pytest.mark.parametrize('dynamic_class',
                         [ProgressiveDynamicConv2d, CenterCropDynamicConv2d])
@pytest.mark.parametrize('mutate_kernel_size', [True, False])
def test_kernel_dynamic_conv2d(dynamic_class: nn.Module,
                               mutate_kernel_size: bool) -> None:

    mutable_in_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    mutable_out_channels = OneShotMutableChannel(
        10, candidate_choices=[4, 8, 10], candidate_mode='number')
    mutable_kernel_size = OneShotMutableValue(value_list=[3, 5, 7])

    with pytest.raises(ValueError):
        d_conv2d = dynamic_class(
            in_channels=10,
            out_channels=10,
            groups=10,
            kernel_size=3,
            stride=1,
            bias=True)
        d_conv2d.mutate_kernel_size(mutable_kernel_size)

    d_conv2d = dynamic_class(
        in_channels=10,
        out_channels=10,
        groups=1,
        kernel_size=7,
        stride=1,
        bias=True)
    d_conv2d.mutate_in_channels(mutable_in_channels)
    d_conv2d.mutate_out_channels(mutable_out_channels)
    if mutate_kernel_size:
        d_conv2d.mutate_kernel_size(mutable_kernel_size)

    with pytest.raises(RuntimeError):
        d_conv2d.to_static_op()

    d_conv2d.mutable_in.current_choice = 8
    d_conv2d.mutable_out.current_choice = 8
    if mutate_kernel_size:
        d_conv2d.mutable_kernel_size.current_choice = 5

    x = torch.rand(10, 8, 224, 224)
    out1 = d_conv2d(x)
    assert out1.size(1) == 8

    fix_mutables = export_fix_subnet(d_conv2d)
    print(fix_mutables)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_conv2d, fix_mutables)
    _fix_dynamic_op(d_conv2d)

    s_conv2d = d_conv2d.to_static_op()
    assert s_conv2d.weight.size(0) == 8
    assert s_conv2d.weight.size(1) == 8
    assert s_conv2d.bias.size(0) == 8
    if mutate_kernel_size:
        assert s_conv2d.kernel_size == (5, 5)
        assert tuple(s_conv2d.weight.shape[2:]) == (5, 5)
    out2 = s_conv2d(x)

    assert torch.equal(out1, out2)
