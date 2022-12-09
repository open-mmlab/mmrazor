# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys

import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models import *  # noqa: F401,F403
from mmrazor.models.mutables import *  # noqa: F401,F403
from mmrazor.registry import MODELS

sys.path.append('tests/test_models/test_architectures/test_backbones')
from utils import MockMutable  # noqa: E402

_FIRST_STAGE_MUTABLE = dict(type='MockMutable', choices=['c1'])
_OTHER_STAGE_MUTABLE = dict(
    type='MockMutable', choices=['c1', 'c2', 'c3', 'c4'])
ARCHSETTING_CFG = [
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride, mutable cfg.
    [16, 1, 1, _FIRST_STAGE_MUTABLE],
    [24, 2, 2, _OTHER_STAGE_MUTABLE],
    [32, 3, 2, _OTHER_STAGE_MUTABLE],
    [64, 4, 2, _OTHER_STAGE_MUTABLE],
    [96, 3, 1, _OTHER_STAGE_MUTABLE],
    [160, 3, 2, _OTHER_STAGE_MUTABLE],
    [320, 1, 1, _OTHER_STAGE_MUTABLE]
]
NORM_CFG = dict(type='BN')
BACKBONE_CFG = dict(
    type='mmrazor.SearchableMobileNetV2',
    first_channels=32,
    last_channels=1280,
    widen_factor=1.0,
    norm_cfg=NORM_CFG,
    arch_setting=ARCHSETTING_CFG)


def test_searchable_mobilenet_mutable() -> None:
    backbone = MODELS.build(BACKBONE_CFG)

    choices = ['c1', 'c2', 'c3', 'c4']
    mutable_nums = 0

    for name, module in backbone.named_modules():
        if isinstance(module, MockMutable):
            if 'layer1' in name:
                assert module.choices == ['c1']
            else:
                assert module.choices == choices
            mutable_nums += 1

    arch_setting = backbone.arch_setting
    target_mutable_nums = 0
    for layer_cfg in arch_setting:
        target_mutable_nums += layer_cfg[1]
    assert mutable_nums == target_mutable_nums


def test_searchable_mobilenet_train() -> None:
    backbone = MODELS.build(BACKBONE_CFG)
    backbone.train(mode=True)
    for m in backbone.modules():
        assert m.training

    backbone.norm_eval = True
    backbone.train(mode=True)
    for m in backbone.modules():
        if isinstance(m, _BatchNorm):
            assert not m.training
        else:
            assert m.training

    x = torch.rand(10, 3, 224, 224)
    assert len(backbone(x)) == 1

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['frozen_stages'] = 5
    backbone = MODELS.build(backbone_cfg)
    backbone.train()

    for param in backbone.conv1.parameters():
        assert not param.requires_grad
    for i in range(1, 8):
        layer = getattr(backbone, f'layer{i}')
        for m in layer.modules():
            if i <= 5:
                assert not m.training
            else:
                assert m.training
        for param in layer.parameters():
            if i <= 5:
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
    assert backbone.out_channel == 1920
