# Copyright (c) OpenMMLab. All rights reserved.
import os
from copy import deepcopy
from os.path import dirname

import mmcv
import numpy as np
import torch
from mmcv import Config, ConfigDict

from mmrazor.models.builder import ALGORITHMS


def _demo_mm_inputs(input_shape=(1, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'img': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def test_autoslim_pretrain():
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

    distiller_cfg = dict(
        type='SelfDistiller',
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd',
                        tau=1,
                        loss_weight=1,
                    )
                ]),
        ])

    algorithm_cfg = ConfigDict(
        type='AutoSlim',
        architecture=architecture_cfg,
        pruner=pruner_cfg,
        distiller=distiller_cfg)

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    model = ALGORITHMS.build(algorithm_cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = model.train_step({'img': imgs, 'gt_label': label}, optimizer)

    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test forward
    losses = model(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0


def test_autoslim_retrain():
    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='MobileNetV2', widen_factor=1.5),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1920,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    pruner_cfg = dict(
        type='RatioPruner',
        ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0))

    root_path = dirname(dirname(dirname(__file__)))
    channel_cfg = [
        os.path.join(root_path, 'data/AUTOSLIM_MBV2_530M_OFFICIAL.yaml'),
        os.path.join(root_path, 'data/AUTOSLIM_MBV2_320M_OFFICIAL.yaml')
    ]

    algorithm_cfg = ConfigDict(
        type='AutoSlim',
        architecture=architecture_cfg,
        pruner=pruner_cfg,
        retraining=True,
        channel_cfg=channel_cfg)

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    # test multi subnet retraining
    model = ALGORITHMS.build(algorithm_cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = model.train_step({'img': imgs, 'gt_label': label}, optimizer)

    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test single subnet retraining
    algorithm_cfg.channel_cfg = algorithm_cfg.channel_cfg[0]
    model = ALGORITHMS.build(algorithm_cfg)
    assert model.deployed
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = model.train_step({'img': imgs, 'gt_label': label}, optimizer)

    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16


def test_spos():

    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='SearchableShuffleNetV2', widen_factor=1.0),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=1024,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
    )

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    mutator_cfg = dict(
        type='OneShotMutator',
        placeholder_mapping=dict(
            all_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    shuffle_3x3=dict(type='ShuffleBlock', kernel_size=3),
                    shuffle_5x5=dict(type='ShuffleBlock', kernel_size=5),
                    shuffle_7x7=dict(type='ShuffleBlock', kernel_size=7),
                    shuffle_xception=dict(type='ShuffleXception'),
                ))))

    algorithm_cfg = dict(
        type='SPOS',
        architecture=architecture_cfg,
        mutator=mutator_cfg,
        retraining=False,
    )

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))
    spos_subnet_path = os.path.join(
        dirname(dirname(dirname(__file__))), 'data/spos_subnet.yaml')

    algorithm_cfg_ = deepcopy(algorithm_cfg)
    algorithm = ALGORITHMS.build(algorithm_cfg_)

    # test forward
    losses = algorithm(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    # test get_subnet_flops
    subnet_dict = algorithm.mutator.sample_subnet()
    flops_supernet = algorithm.get_subnet_flops()
    algorithm.mutator.set_subnet(subnet_dict)
    flops_subnet = algorithm.get_subnet_flops()
    assert flops_supernet > flops_subnet > 0

    # deploy_subnet in BaseMutator
    spos_subnet_dict = mmcv.fileio.load(spos_subnet_path)
    algorithm.mutator.deploy_subnet(algorithm.architecture, spos_subnet_dict)
    flops_subnet_spos = algorithm.get_subnet_flops()
    assert flops_supernet > flops_subnet_spos > 0


def test_spos_mb():

    model_cfg = dict(
        type='mmcls.ImageClassifier',
        backbone=dict(type='SearchableMobileNet', widen_factor=1.0),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=1280,
            loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
    )

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=model_cfg,
    )

    mutator_cfg = dict(
        type='OneShotMutator',
        placeholder_mapping=dict(
            searchable_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    mbv2_k3e3=dict(
                        type='MBBlock',
                        kernel_size=3,
                        expand_ratio=3,
                        act_cfg=dict(type='ReLU6')),
                    mbv2_k5e3=dict(
                        type='MBBlock',
                        kernel_size=5,
                        expand_ratio=3,
                        act_cfg=dict(type='ReLU6')),
                    mbv2_k7e3=dict(
                        type='MBBlock',
                        kernel_size=7,
                        expand_ratio=3,
                        act_cfg=dict(type='ReLU6')),
                    mbv2_k3e6=dict(
                        type='MBBlock',
                        kernel_size=3,
                        expand_ratio=6,
                        act_cfg=dict(type='ReLU6')),
                    mbv2_k5e6=dict(
                        type='MBBlock',
                        kernel_size=5,
                        expand_ratio=6,
                        act_cfg=dict(type='ReLU6')),
                    mbv2_k7e6=dict(
                        type='MBBlock',
                        kernel_size=7,
                        expand_ratio=6,
                        act_cfg=dict(type='ReLU6')),
                    identity=dict(type='Identity'))),
            first_blocks=dict(
                type='OneShotOP',
                choices=dict(
                    mbv2_k3e1=dict(
                        type='MBBlock',
                        kernel_size=3,
                        expand_ratio=1,
                        act_cfg=dict(type='ReLU6')), ))))

    algorithm_cfg = dict(
        type='SPOS',
        architecture=architecture_cfg,
        mutator=mutator_cfg,
        retraining=False,
    )

    imgs = torch.randn(16, 3, 224, 224)
    label = torch.randint(0, 1000, (16, ))

    algorithm_cfg_ = deepcopy(algorithm_cfg)
    algorithm = ALGORITHMS.build(algorithm_cfg_)

    # test forward
    losses = algorithm(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0


def test_detnas():
    config_path = os.path.join(
        dirname(dirname(dirname(__file__))),
        'data/detnas_frcnn_shufflenet_fpn.py')
    config = Config.fromfile(config_path)

    # test detnas init
    algorithm = ALGORITHMS.build(config.algorithm)
    assert hasattr(algorithm, 'architecture')


def test_darts():

    architecture_cfg = dict(
        type='MMClsArchitecture',
        model=dict(
            type='mmcls.ImageClassifier',
            backbone=dict(
                type='DartsBackbone',
                in_channels=3,
                base_channels=8,
                num_layers=4,
                num_nodes=4,
                stem_multiplier=3,
                out_indices=(3, )),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=10,
                in_channels=128,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ),
        ),
    )

    mutator_cfg = dict(
        type='DartsMutator',
        placeholder_mapping=dict(
            node=dict(
                type='DifferentiableOP',
                with_arch_param=True,
                choices=dict(
                    zero=dict(type='DartsZero'),
                    skip_connect=dict(
                        type='DartsSkipConnect',
                        norm_cfg=dict(type='BN', affine=False)),
                    max_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='max',
                        norm_cfg=dict(type='BN', affine=False)),
                    avg_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='avg',
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_3x3=dict(
                        type='DartsSepConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_5x5=dict(
                        type='DartsSepConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_3x3=dict(
                        type='DartsDilConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_5x5=dict(
                        type='DartsDilConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                )),
            node_edge=dict(
                type='DifferentiableEdge',
                num_chosen=2,
                with_arch_param=False,
            )),
    )

    algorithm_cfg = dict(
        type='Darts',
        architecture=architecture_cfg,
        mutator=mutator_cfg,
        retraining=False,
        unroll=False,
    )

    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    algorithm_cfg_ = deepcopy(algorithm_cfg)
    algorithm = ALGORITHMS.build(algorithm_cfg_)

    optimizers = dict(
        architecture=torch.optim.SGD(
            algorithm.architecture.parameters(), lr=0.01),
        mutator=torch.optim.SGD(algorithm.mutator.parameters(), lr=0.01),
    )

    data = [{'img': imgs, 'gt_label': label}, {'img': imgs, 'gt_label': label}]
    # test forward
    losses = algorithm.train_step(data, optimizers)
    assert losses['loss'].item() > 0


def test_cwd():
    config_path = './tests/data/cwd_pspnet.py'

    config = Config.fromfile(config_path)

    mm_inputs = _demo_mm_inputs(input_shape=(16, 3, 256, 256), num_classes=19)
    algorithm = ALGORITHMS.build(config.algorithm)

    # test algorithm train_step
    losses = algorithm.train_step(mm_inputs, None)
    assert losses['loss'].item() > 0
