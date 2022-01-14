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

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner = PRUNERS.build(pruner_cfg)

    pruner.prepare_from_supernet(architecture)
    assert hasattr(pruner, 'channel_spaces')

    # test set_min_channel
    pruner_cfg_ = deepcopy(pruner_cfg)
    pruner_cfg_['ratios'].insert(0, 0)
    pruner_ = PRUNERS.build(pruner_cfg_)
    architecture_ = ARCHITECTURES.build(architecture_cfg)
    pruner_.prepare_from_supernet(architecture_)
    with pytest.raises(AssertionError):
        # Output channels should be a positive integer not zero
        pruner_.set_min_channel()

    # test set_max_channel
    pruner.set_max_channel()
    for name, module in architecture.model.named_modules():
        if hasattr(module, 'in_mask'):
            assert module.in_mask.sum() == module.in_mask.numel()
        if hasattr(module, 'out_mask'):
            assert module.out_mask.sum() == module.out_mask.numel()

    # test channel bins
    pruner.set_min_channel()
    channel_bins_dict = pruner.get_max_channel_bins(max_channel_bins=4)
    pruner.set_channel_bins(channel_bins_dict, 4)
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


def test_bcnet_pruner():
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='MobileNetV2', widen_factor=1.5),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1920,
            loss=dict(
                type='LabelSmoothLoss',
                mode='original',
                label_smooth_val=0.1,
                loss_weight=1.0),
            topk=(1, 5),
        ))

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    pruner_cfg = dict(
        type='BCNetPruner',
        ratios=(1 / 20, 2 / 20, 3 / 20, 4 / 20, 5 / 20, 6 / 20, 7 / 20, 8 / 20,
                9 / 20, 10 / 20, 11 / 20, 12 / 20, 13 / 20, 14 / 20, 15 / 20,
                16 / 20, 17 / 20, 18 / 20, 19 / 20, 1.0))

    architecture = ARCHITECTURES.build(architecture_cfg)
    pruner = PRUNERS.build(pruner_cfg)

    pruner.prepare_from_supernet(architecture)
    assert hasattr(pruner, 'channel_spaces')

    # test reverse_subnet
    subnet_dict = pruner.sample_subnet()
    reverse_once = pruner.reverse_subnet(subnet_dict)
    reverse_twice = pruner.reverse_subnet(reverse_once)
    for space_id in subnet_dict.keys():
        assert torch.equal(subnet_dict[space_id], reverse_twice[space_id])

    # test get_complementary_subnet
    complementary_subnet = pruner.get_complementary_subnet(subnet_dict)
    for space_id in subnet_dict.keys():
        assert subnet_dict[space_id].sum() + complementary_subnet[
            space_id].sum() == subnet_dict[space_id].numel()
