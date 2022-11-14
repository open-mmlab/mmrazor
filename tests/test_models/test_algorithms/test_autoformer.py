# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch

from mmrazor.models import Autoformer
from mmrazor.registry import MODELS

arch_setting = dict(
    mlp_ratios=[3.0, 3.5, 4.0],
    num_heads=[8, 9, 10],
    depth=[14, 15, 16],
    embed_dims=[528, 576, 624])

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

ARCHITECTURE_CFG = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    backbone=dict(
        _scope_='mmrazor',
        type='AutoformerBackbone',
        arch_setting=arch_setting),
    neck=None,
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=624,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            mode='original',
            num_classes=1000,
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5)),
    connect_head=dict(connect_with_backbone='backbone.last_mutable'),
)

ALGORITHM_CFG = dict(
    type='mmrazor.Autoformer',
    architecture=ARCHITECTURE_CFG,
    fix_subnet=None,
    mutators=dict(
        channel_mutator=dict(
            type='mmrazor.OneShotChannelMutator',
            channel_unit_cfg={
                'type': 'OneShotMutableChannelUnit',
                'default_args': {
                    'unit_predefined': True
                }
            },
            parse_cfg={'type': 'Predefined'}),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')))


class TestAUTOFORMER(TestCase):

    def test_init(self):
        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate autoformer with built `algorithm`.
        autoformer_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(autoformer_algo, Autoformer)
        # autoformer mutators include channel_mutator and value_mutator
        assert 'channel_mutator' in autoformer_algo.mutators
        assert 'value_mutator' in autoformer_algo.mutators

        # autoformer search_groups
        random_subnet = autoformer_algo.sample_subnet()
        self.assertIsInstance(random_subnet, dict)

        # autoformer_algo support training
        self.assertTrue(autoformer_algo.is_supernet)

        # initiate autoformer without any `mutator`.
        ALGORITHM_CFG_SUPERNET.pop('type')
        ALGORITHM_CFG_SUPERNET['mutators'] = None
        with self.assertRaisesRegex(
                AssertionError,
                'mutator cannot be None when fix_subnet is None.'):
            _ = Autoformer(**ALGORITHM_CFG_SUPERNET)

        # initiate autoformer with error type `mutator`.
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
                                    'autoformer only support predefined.'):
            _ = Autoformer(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # supernet
        inputs = torch.randn(1, 3, 224, 224)
        autoformer = MODELS.build(ALGORITHM_CFG)
        loss = autoformer(inputs)
        assert loss.size(1) == 1000
