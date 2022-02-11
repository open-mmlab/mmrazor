# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmrazor.models.builder import ARCHITECTURES, PRUNERS


def test_ratio_pruner():
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(
            type='mmcls.ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    pruner_cfg = dict(
        type='RatioPruner',
        ratios=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0])

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    pruner_cfg_ = deepcopy(pruner_cfg)
    pruner_cfg_['ratios'].insert(0, 0)
    with pytest.raises(AssertionError):
        # All the numbers in ``ratios`` should be positive
        PRUNERS.build(pruner_cfg_)

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner = PRUNERS.build(pruner_cfg)

    pruner.prepare_from_supernet(architecture)
    assert hasattr(pruner, 'search_spaces')

    pruner.set_min_channel()
    pruner.set_max_channel()
    for name, module in architecture.model.named_modules():
        if hasattr(module, 'in_mask'):
            assert module.in_mask.sum() == module.in_mask.numel()
        if hasattr(module, 'out_mask'):
            assert module.out_mask.sum() == module.out_mask.numel()

    # test making groups logic
    subnet_dict = pruner.sample_subnet()
    assert isinstance(subnet_dict, dict)
    pruner.set_subnet(subnet_dict)
    subnet_dict = pruner.export_subnet()
    assert isinstance(subnet_dict, dict)
    pruner.deploy_subnet(architecture, subnet_dict)
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0
