# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from mmrazor.models.mutables import SquentialMutableChannel
from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet
from ..utils import fix_dynamic_op

from mmrazor.models.architectures.dynamic_ops import (  # isort:skip
    DynamicLinear, DynamicLinearMixin)


@pytest.mark.parametrize('bias', [True, False])
def test_dynamic_linear(bias) -> None:
    mutable_in_features = SquentialMutableChannel(10)
    mutable_out_features = SquentialMutableChannel(10)

    d_linear = DynamicLinear(in_features=10, out_features=10, bias=bias)

    mock_mutable = MagicMock()
    with pytest.raises(ValueError):
        d_linear.register_mutable_attr('in_features', mock_mutable)
    with pytest.raises(ValueError):
        d_linear.register_mutable_attr('out_features', mock_mutable)

    mock_mutable.current_mask = torch.rand(8)
    with pytest.raises(ValueError):
        d_linear.register_mutable_attr('in_features', mock_mutable)
    with pytest.raises(ValueError):
        d_linear.register_mutable_attr('out_features', mock_mutable)

    d_linear.register_mutable_attr('in_features', mutable_in_features)
    d_linear.register_mutable_attr('out_features', mutable_out_features)

    with pytest.raises(RuntimeError):
        d_linear.to_static_op()

    d_linear.get_mutable_attr('in_channels').current_choice = 8
    d_linear.get_mutable_attr('out_channels').current_choice = 4

    x = torch.rand(10, 8)
    out1 = d_linear(x)
    assert out1.size(1) == 4

    with pytest.raises(RuntimeError):
        _ = d_linear.to_static_op()

    fix_mutables = export_fix_subnet(d_linear)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_linear, fix_mutables)
    fix_dynamic_op(d_linear, fix_mutables)
    assert isinstance(d_linear, nn.Linear)
    assert isinstance(d_linear, DynamicLinearMixin)

    s_linear = d_linear.to_static_op()
    assert s_linear.weight.size(0) == 4
    assert s_linear.weight.size(1) == 8
    if bias:
        assert s_linear.bias.size(0) == 4
    assert not isinstance(s_linear, DynamicLinearMixin)
    assert isinstance(s_linear, nn.Linear)
    out2 = s_linear(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize(
    ['is_mutate_in_features', 'in_features', 'out_features'], [(True, 6, 10),
                                                               (False, 10, 4),
                                                               (None, 10, 10)])
def test_dynamic_linear_mutable_single_features(
        is_mutate_in_features: Optional[bool], in_features: int,
        out_features: int) -> None:
    d_linear = DynamicLinear(in_features=10, out_features=10, bias=True)
    mutable_channels = SquentialMutableChannel(10)

    if is_mutate_in_features is not None:
        if is_mutate_in_features:
            d_linear.register_mutable_attr('in_channels', mutable_channels)
        else:
            d_linear.register_mutable_attr('out_channels', mutable_channels)

    if is_mutate_in_features:
        d_linear.get_mutable_attr('in_channels').current_choice = in_features
        assert d_linear.get_mutable_attr('out_channels') is None
    elif is_mutate_in_features is False:
        d_linear.get_mutable_attr('out_channels').current_choice = out_features
        assert d_linear.get_mutable_attr('in_channels') is None

    x = torch.rand(3, in_features)
    out1 = d_linear(x)

    assert out1.size(1) == out_features

    if is_mutate_in_features is not None:
        with pytest.raises(RuntimeError):
            _ = d_linear.to_static_op()

    fix_mutables = export_fix_subnet(d_linear)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_linear, fix_mutables)
    fix_dynamic_op(d_linear, fix_mutables)

    s_linear = d_linear.to_static_op()
    assert s_linear.weight.size(0) == out_features
    assert s_linear.weight.size(1) == in_features
    out2 = s_linear(x)

    assert torch.equal(out1, out2)
