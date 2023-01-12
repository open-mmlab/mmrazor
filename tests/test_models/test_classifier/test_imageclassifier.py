# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models import SearchableImageClassifier


class TestSearchableImageClassifier(TestCase):

    def test_init(self):

        arch_setting = dict(
            mlp_ratios=[3.0, 3.5, 4.0],
            num_heads=[8, 9, 10],
            depth=[14, 15, 16],
            embed_dims=[528, 576, 624])

        supernet_kwargs = dict(
            backbone=dict(
                _scope_='mmrazor',
                type='AutoformerBackbone',
                arch_setting=arch_setting),
            neck=None,
            head=dict(
                _scope_='mmrazor',
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

        supernet = SearchableImageClassifier(**supernet_kwargs)

        # test connect_with_backbone
        self.assertEqual(
            supernet.backbone.last_mutable.activated_channels,
            len(
                supernet.head.fc.get_mutable_attr(
                    'in_channels').current_choice))
