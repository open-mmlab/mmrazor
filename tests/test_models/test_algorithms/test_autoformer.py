# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch

from mmrazor.models import Autoformer, NasMutator
from mmrazor.registry import MODELS

arch_setting = dict(
    mlp_ratios=[3.0, 3.5, 4.0],
    num_heads=[8, 9, 10],
    depth=[14, 15, 16],
    embed_dims=[528, 576, 624])

MUTATOR_CFG = dict(type='NasMutator')

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
    mutator=MUTATOR_CFG)


class TestAutoFormer(TestCase):

    def test_init(self):
        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate autoformer with built `algorithm`.
        autoformer_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(autoformer_algo, Autoformer)
        self.assertIsInstance(autoformer_algo.mutator, NasMutator)

        # autoformer search_groups
        random_subnet = autoformer_algo.mutator.sample_choices()
        self.assertIsInstance(random_subnet, dict)

        # initiate autoformer without any `mutator`.
        ALGORITHM_CFG_SUPERNET.pop('type')
        ALGORITHM_CFG_SUPERNET['mutator'] = None
        none_type = type(ALGORITHM_CFG_SUPERNET['mutator'])
        with self.assertRaisesRegex(
                TypeError, 'mutator should be a `dict` or `NasMutator` '
                f'instance, but got {none_type}.'):
            _ = Autoformer(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # supernet
        inputs = torch.randn(1, 3, 224, 224)
        autoformer = MODELS.build(ALGORITHM_CFG)
        loss = autoformer(inputs)
        assert loss.size(1) == 1000
