# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import sys
import tempfile

import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models import *  # noqa: F401,F403
from mmrazor.models.mutables import *  # noqa: F401,F403
from mmrazor.registry import MODELS

sys.path.append('tests/test_models/test_architectures/test_backbones')
from utils import MockMutable  # noqa: E402

STAGE_MUTABLE = dict(type='MockMutable', choices=['c1', 'c2', 'c3', 'c4'])
ARCHSETTING_CFG = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, STAGE_MUTABLE],
    [160, 4, STAGE_MUTABLE],
    [320, 8, STAGE_MUTABLE],
    [640, 4, STAGE_MUTABLE],
]

NORM_CFG = dict(type='BN')
BACKBONE_CFG = dict(
    type='mmrazor.SearchableShuffleNetV2',
    widen_factor=1.0,
    norm_cfg=NORM_CFG,
    arch_setting=ARCHSETTING_CFG)


def test_searchable_shufflenet_v2_mutable() -> None:
    backbone = MODELS.build(BACKBONE_CFG)

    choices = ['c1', 'c2', 'c3', 'c4']
    mutable_nums = 0

    for module in backbone.modules():
        if isinstance(module, MockMutable):
            assert module.choices == choices
            mutable_nums += 1

    arch_setting = backbone.arch_setting
    target_mutable_nums = 0
    for layer_cfg in arch_setting:
        target_mutable_nums += layer_cfg[1]
    assert mutable_nums == target_mutable_nums


def test_searchable_shufflenet_v2_train() -> None:
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
    backbone_cfg['frozen_stages'] = 2
    backbone = MODELS.build(backbone_cfg)
    backbone.train()

    for param in backbone.conv1.parameters():
        assert not param.requires_grad
    for i in range(2):
        layer = backbone.layers[i]
        for m in layer.modules():
            if i < 2:
                assert not m.training
            else:
                assert m.training
        for param in layer.parameters():
            if i < 2:
                assert not param.requires_grad
            else:
                assert param.requires_grad


def test_searchable_shufflenet_v2_init() -> None:
    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['out_indices'] = (5, )

    with pytest.raises(ValueError):
        MODELS.build(backbone_cfg)

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['frozen_stages'] = 5

    with pytest.raises(ValueError):
        MODELS.build(backbone_cfg)

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['with_last_layer'] = False
    with pytest.raises(ValueError):
        MODELS.build(backbone_cfg)

    backbone_cfg['out_indices'] = (3, )
    backbone = MODELS.build(backbone_cfg)
    assert len(backbone.layers) == 4


def test_searchable_shufflenet_v2_init_weights() -> None:
    backbone = MODELS.build(BACKBONE_CFG)
    backbone.init_weights()

    for m in backbone.modules():
        if isinstance(m, (_BatchNorm, GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                assert torch.equal(m.weight, torch.ones_like(m.weight))
            if hasattr(m, 'bias') and m.bias is not None:
                bias_tensor = torch.ones_like(m.bias)
                bias_tensor *= 0.0001
                assert torch.equal(bias_tensor, m.bias)

    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone = MODELS.build(backbone_cfg)
    torch.save(backbone.state_dict(), checkpoint_path)
    backbone_cfg['init_cfg'] = dict(
        type='Pretrained', checkpoint=checkpoint_path)
    backbone = MODELS.build(backbone_cfg)

    name2weight = dict()
    for name, m in backbone.named_modules():
        if isinstance(m, (_BatchNorm, GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                name2weight[name] = m.weight.clone()

    backbone.init_weights()
    for name, m in backbone.named_modules():
        if isinstance(m, (_BatchNorm, GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                if name in name2weight:
                    assert torch.equal(name2weight[name], m.weight)

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['norm_cfg'] = dict(type='BN', track_running_stats=False)
    backbone = MODELS.build(backbone_cfg)
    backbone.init_weights()

    backbone_cfg = copy.deepcopy(BACKBONE_CFG)
    backbone_cfg['norm_cfg'] = dict(type='GN', num_groups=1)
    backbone = MODELS.build(backbone_cfg)
    backbone.init_weights()
