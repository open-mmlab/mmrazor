# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from copy import deepcopy

import numpy as np
import torch

from mmrazor.models.builder import ARCHITECTURES


def test_architecture_mmcls():
    model_cfg = dict(
        dict(
            type='mmcls.ImageClassifier',
            backbone=dict(
                type='mmcls.ResNet_CIFAR',
                depth=50,
                num_stages=4,
                out_indices=(3, ),
                style='pytorch'),
            neck=dict(type='mmcls.GlobalAveragePooling'),
            head=dict(
                type='mmcls.LinearClsHead',
                num_classes=10,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss'))), )

    architecture_cfg = dict(type='MMClsArchitecture', model=model_cfg)

    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    supernet_cfg_ = deepcopy(architecture_cfg)
    architecture = ARCHITECTURES.build(supernet_cfg_)

    # test property
    assert architecture.model.with_neck
    assert architecture.model.with_head

    # test train_step
    outputs = architecture.model.train_step({
        'img': imgs,
        'gt_label': label
    }, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test val_step
    outputs = architecture.model.val_step({
        'img': imgs,
        'gt_label': label
    }, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test forward
    losses = architecture(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    # test forward_test
    architecture_cfg_ = deepcopy(architecture_cfg)
    architecture = ARCHITECTURES.build(architecture_cfg_)
    pred = architecture(imgs, return_loss=False, img_metas=None)
    assert isinstance(pred, list) and len(pred) == 16

    single_img = torch.randn(1, 3, 32, 32)
    pred = architecture(single_img, return_loss=False, img_metas=None)
    assert isinstance(pred, list) and len(pred) == 1

    # test simple_test
    single_img = torch.randn(1, 3, 32, 32)
    pred = architecture.simple_test(single_img, img_metas=None)
    assert isinstance(pred, list) and len(pred) == 1

    # test show_result
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    result = dict(pred_class='cat', pred_label=0, pred_score=0.9)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = osp.join(tmpdir, 'out.png')
        architecture.show_result(img, result, out_file=out_file)
        assert osp.exists(out_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = osp.join(tmpdir, 'out.png')
        architecture.show_result(img, result, out_file=out_file)
        assert osp.exists(out_file)
