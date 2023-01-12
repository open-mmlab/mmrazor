# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch

from mmrazor.models import BigNAS
from mmrazor.registry import MODELS

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

MUTATOR_CFG = dict(
    channel_mutator=dict(
        type='mmrazor.OneShotChannelMutator',
        channel_unit_cfg={
            'type': 'OneShotMutableChannelUnit',
            'default_args': {
                'unit_predefined': True
            }
        },
        parse_cfg={'type': 'Predefined'}),
    value_mutator=dict(type='mmrazor.DynamicValueMutator'))

DISTILLER_CFG = dict(
    _scope_='mmrazor',
    type='ConfigurableDistiller',
    teacher_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    student_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    distill_losses=dict(
        loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
    loss_forward_mappings=dict(
        loss_kl=dict(
            preds_S=dict(recorder='fc', from_student=True),
            preds_T=dict(recorder='fc', from_student=False))))

ARCHITECTURE_CFG = dict(
    type='mmrazor.SearchableImageClassifier',
    backbone=dict(
        type='mmrazor.AttentiveMobileNetV3',
        arch_setting=arch_setting,
        out_indices=(4, ),
        conv_cfg=dict(type='mmrazor.BigNasConv2d'),
        norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.0)),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        _scope_='mmrazor',
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=72,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            mode='original',
            num_classes=1000,
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5)),
    connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'),
)

ALGORITHM_CFG = dict(
    type='mmrazor.BigNAS',
    architecture=ARCHITECTURE_CFG,
    mutators=MUTATOR_CFG,
    distiller=DISTILLER_CFG)


class TestBigNAS(TestCase):

    def test_init(self):
        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate bignas with built `algorithm`.
        bignas_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(bignas_algo, BigNAS)
        # bignas mutators include channel_mutator and value_mutator
        assert 'channel_mutator' in bignas_algo.mutators
        assert 'value_mutator' in bignas_algo.mutators

        # bignas search_groups
        random_subnet = bignas_algo.sample_subnet()
        self.assertIsInstance(random_subnet, dict)

        # bignas_algo support training
        self.assertTrue(bignas_algo.is_supernet)

        # initiate bignas without any `mutator`.
        ALGORITHM_CFG_SUPERNET.pop('type')
        ALGORITHM_CFG_SUPERNET['mutators'] = None
        none_type = type(ALGORITHM_CFG_SUPERNET['mutators'])
        with self.assertRaisesRegex(
                TypeError, f'mutator should be a `dict` but got {none_type}'):
            _ = BigNAS(**ALGORITHM_CFG_SUPERNET)

        # initiate bignas with error type `mutator`.
        backwardtracer_cfg = dict(
            type='OneShotChannelMutator',
            channel_unit_cfg=dict(
                type='OneShotMutableChannelUnit',
                default_args=dict(
                    candidate_choices=list(i / 12 for i in range(2, 13)),
                    choice_mode='ratio')),
            parse_cfg=dict(
                type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')))
        ALGORITHM_CFG_SUPERNET['mutators'] = dict(
            channel_mutator=backwardtracer_cfg,
            value_mutator=dict(type='mmrazor.DynamicValueMutator'))
        with self.assertRaisesRegex(AssertionError,
                                    'BigNAS only support predefined.'):
            _ = BigNAS(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # supernet
        inputs = torch.randn(1, 3, 224, 224)
        bignas = MODELS.build(ALGORITHM_CFG)
        loss = bignas(inputs)
        assert loss.size(1) == 1000
