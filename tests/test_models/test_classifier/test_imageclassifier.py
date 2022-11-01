# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmrazor.models import SearchableImageClassifier


class TestSearchableImageClassifier(TestCase):

    def test_init(self):

        supernet_kwargs = dict(
            backbone=dict(_scope_='mmrazor', type='AutoformerBackbone'),
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
        )

        _ = SearchableImageClassifier(**supernet_kwargs)

        # test input_resizer_cfg
        # TODO

        # test connect_with_backbone
        # TODO
