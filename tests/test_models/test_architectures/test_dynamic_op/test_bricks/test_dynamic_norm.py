# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Type
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from mmrazor.models.architectures.dynamic_ops import (DynamicBatchNorm1d,
                                                      DynamicBatchNorm2d,
                                                      DynamicBatchNorm3d,
                                                      DynamicMixin)
from mmrazor.models.mutables import SquentialMutableChannel
from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet
from ..utils import fix_dynamic_op


@pytest.mark.parametrize('dynamic_class,input_shape',
                         [(DynamicBatchNorm1d, (10, 8, 224)),
                          (DynamicBatchNorm2d, (10, 8, 224, 224)),
                          (DynamicBatchNorm3d, (10, 8, 3, 224, 224))])
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('track_running_stats', [True, False])
def test_dynamic_bn(dynamic_class: Type[nn.modules.batchnorm._BatchNorm],
                    input_shape: Tuple[int], affine: bool,
                    track_running_stats: bool) -> None:
    mutable_num_features = SquentialMutableChannel(10)

    d_bn = dynamic_class(
        num_features=10,
        affine=affine,
        track_running_stats=track_running_stats)
    if not affine and not track_running_stats:
        with pytest.raises(RuntimeError):
            d_bn.register_mutable_attr('num_features', mutable_num_features)
    else:
        mock_mutable = MagicMock()
        with pytest.raises(ValueError):
            d_bn.register_mutable_attr('num_features', mock_mutable)
        mock_mutable.current_mask = torch.rand(5)
        with pytest.raises(ValueError):
            d_bn.register_mutable_attr('num_features', mock_mutable)

        d_bn.register_mutable_attr('num_features', mutable_num_features)
        assert d_bn.get_mutable_attr('in_channels') is d_bn.get_mutable_attr(
            'out_channels')

    if affine or track_running_stats:
        d_bn.get_mutable_attr('in_channels').current_choice = 8

    with pytest.raises(ValueError):
        wrong_shape_x = torch.rand(8)
        _ = d_bn(wrong_shape_x)

    x = torch.rand(*input_shape)
    out1 = d_bn(x)
    assert out1.size(1) == 8

    fix_mutables = export_fix_subnet(d_bn)
    with pytest.raises(RuntimeError):
        load_fix_subnet(d_bn, fix_mutables)
    fix_dynamic_op(d_bn, fix_mutables)
    assert isinstance(d_bn, dynamic_class)
    assert isinstance(d_bn, DynamicMixin)

    s_bn = d_bn.to_static_op()
    if affine:
        assert s_bn.weight.size(0) == 8
        assert s_bn.bias.size(0) == 8
    if track_running_stats:
        assert s_bn.running_mean.size(0) == 8
        assert s_bn.running_var.size(0) == 8
    assert not isinstance(s_bn, DynamicMixin)
    assert isinstance(s_bn, d_bn.static_op_factory)
    out2 = s_bn(x)

    assert torch.equal(out1, out2)


@pytest.mark.parametrize(['static_class', 'dynamic_class', 'input_shape'],
                         [(nn.BatchNorm1d, DynamicBatchNorm1d, (10, 8, 224)),
                          (nn.BatchNorm2d, DynamicBatchNorm2d,
                           (10, 8, 224, 224)),
                          (nn.BatchNorm3d, DynamicBatchNorm3d,
                           (10, 8, 3, 224, 224))])
def test_bn_track_running_stats(
    static_class: Type[nn.modules.batchnorm._BatchNorm],
    dynamic_class: Type[nn.modules.batchnorm._BatchNorm],
    input_shape: Tuple[int],
) -> None:
    mutable_num_features = SquentialMutableChannel(10)
    mutable_num_features.current_choice = 8
    d_bn = dynamic_class(
        num_features=10, track_running_stats=True, affine=False)
    d_bn.register_mutable_attr('num_features', mutable_num_features)

    s_bn = static_class(num_features=8, track_running_stats=True, affine=False)

    d_bn.train()
    s_bn.train()
    mask = d_bn._get_num_features_mask()
    for _ in range(10):
        x = torch.rand(*input_shape)
        _ = d_bn(x)
        _ = s_bn(x)

        d_running_mean = d_bn.running_mean[mask]
        d_running_var = d_bn.running_var[mask]

        assert torch.equal(s_bn.running_mean, d_running_mean)
        assert torch.equal(s_bn.running_var, d_running_var)

    d_bn.eval()
    s_bn.eval()
    x = torch.rand(*input_shape)

    assert torch.equal(d_bn(x), s_bn(x))
