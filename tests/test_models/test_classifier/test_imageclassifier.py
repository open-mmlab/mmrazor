# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

from mmengine import ConfigDict

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
        supernet_kwargs_ = copy.deepcopy(supernet_kwargs)
        supernet_kwargs_['input_resizer_cfg'] = None
        with self.assertRaisesRegex(AssertionError,
                                    '"loss_toy" is not in distill'):
            _ = SearchableImageClassifier(**supernet_kwargs_)

        # test connect_with_backbone
        supernet_kwargs_ = copy.deepcopy(supernet_kwargs)
        supernet_kwargs_['head'] = dict(
            toy=dict(type='ToyDistillLoss'))
        supernet_kwargs_['loss_forward_mappings'] = dict(
            toy=dict(
                arg1=dict(from_student=True, recorder='conv'),
                arg2=dict(from_student=False, recorder='conv')))
        with self.assertWarnsRegex(UserWarning, 'Warning: If toy is a'):
            _ = SearchableImageClassifier(**supernet_kwargs_)
