# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys

import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicBatchNorm2d,
                                                      DynamicSequential)
from mmrazor.models.mutables import (MutableChannelContainer,
                                     OneShotMutableValue)
from mmrazor.models.utils import parse_values
from mmrazor.registry import MODELS

sys.path.append('tests/test_models/test_architectures/test_backbones')

arch_setting = dict(
    kernel_size=[
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
    ],
    num_blocks=[
        [1, 2, 1],
        [3, 6, 1],
        [3, 6, 1],
        [1, 2, 1],
    ],
    expand_ratio=[
        [1, 1, 1],
        [4, 6, 1],
        [4, 6, 1],
        [4, 6, 1],
        [4, 6, 1],
    ],
    num_out_channels=[
        [16, 24, 8],  # first layer
        [16, 24, 8],
        [24, 32, 8],
        [32, 40, 8],
        [64, 72, 8],
        [72, 72, 8],  # last layer
    ])

BACKBONE_CFG = dict(
    type='mmrazor.AttentiveMobileNetV3',
    arch_setting=arch_setting,
    out_indices=(4, ),
    conv_cfg=dict(type='mmrazor.BigNasConv2d'),
    norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.0))


def test_attentive_mobilenet_mutable() -> None:
    backbone = MODELS.build(BACKBONE_CFG)

    out_channels = backbone.arch_setting['num_out_channels']
    out_channels = parse_values(out_channels)

    for module in backbone.modules():
        if isinstance(module, BigNasConv2d):
            assert isinstance(module.mutable_attrs.in_channels,
                              MutableChannelContainer)
            assert isinstance(module.mutable_attrs.out_channels,
                              MutableChannelContainer)
        elif isinstance(module, DynamicBatchNorm2d):
            assert isinstance(module.mutable_attrs.num_features,
                              MutableChannelContainer)
        elif isinstance(module, DynamicSequential):
            assert isinstance(module.mutable_depth, OneShotMutableValue)

    assert backbone.last_mutable_channels.num_channels == max(out_channels[-1])


def test_attentive_mobilenet_train() -> None:
    backbone = MODELS.build(BACKBONE_CFG)
    backbone.train(mode=True)
    for m in backbone.modules():
        assert m.training

    backbone.norm_eval = True
    backbone.train(mode=True)
    for m in backbone.modules():
        if isinstance(m, _BatchNorm):
            assert not m.training

    x = torch.rand(10, 3, 224, 224)
    assert len(backbone(x)) == 1

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['frozen_stages'] = 2
    backbone = MODELS.build(backbone_cfg)
    backbone.train()

    for param in backbone.first_conv.parameters():
        assert not param.requires_grad
    for i, layer in enumerate(backbone.layers):
        for param in layer.parameters():
            if i <= 1:
                assert not param.requires_grad
            else:
                assert param.requires_grad


def test_searchable_mobilenet_init() -> None:
    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['out_indices'] = (10, )

    with pytest.raises(ValueError):
        MODELS.build(backbone_cfg)

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['frozen_stages'] = 8

    with pytest.raises(ValueError):
        MODELS.build(backbone_cfg)

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['widen_factor'] = 1.5
    backbone = MODELS.build(backbone_cfg)
    assert backbone.out_channels == 112
