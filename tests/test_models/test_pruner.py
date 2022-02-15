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

    _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, False)
    with pytest.raises(AssertionError):
        _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, True)

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


def _test_reset_bn_running_stats(architecture_cfg, pruner_cfg, should_fail):
    import os
    import random

    import numpy as np

    def set_seed(seed: int) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    output_list = []

    def output_hook(self, input, output) -> None:
        output_list.append(output)

    set_seed(1024)

    imgs = torch.randn(16, 3, 224, 224)

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()

    architecture1 = ARCHITECTURES.build(architecture_cfg)
    pruner1 = PRUNERS.build(pruner_cfg)
    if should_fail:
        pruner1._reset_norm_running_stats = lambda *_: None
    set_seed(1)
    pruner1.prepare_from_supernet(architecture1)
    architecture1.model.head.fc.register_forward_hook(output_hook)
    architecture1.eval()
    architecture1(imgs, return_loss=False)

    set_seed(1024)
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(random_rng_state)

    architecture2 = ARCHITECTURES.build(architecture_cfg)
    pruner2 = PRUNERS.build(pruner_cfg)
    if should_fail:
        pruner2._reset_norm_running_stats = lambda *_: None
    set_seed(2)
    pruner2.prepare_from_supernet(architecture2)
    architecture2.model.head.fc.register_forward_hook(output_hook)
    architecture2.eval()
    architecture2(imgs, return_loss=False)

    assert torch.equal(output_list[0].norm(p='fro'),
                       output_list[1].norm(p='fro'))
