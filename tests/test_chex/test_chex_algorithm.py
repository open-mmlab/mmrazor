# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.logging import MessageHub

from mmrazor.models.chex import ChexAlgorithm

MODEL_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


class TestChexAlgorithm(unittest.TestCase):

    def test_chex_algorithm(self):
        algorithm = ChexAlgorithm(MODEL_CFG)
        x = torch.rand([2, 3, 64, 64])
        self._set_epoch_ite(0, 2, 100)
        _ = algorithm(x)

    def _set_epoch_ite(self, epoch, ite, max_epoch):
        iter_per_epoch = 10
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('epoch', epoch)
        message_hub.update_info('max_epochs', max_epoch)
        message_hub.update_info('max_iters', max_epoch * 10)
        message_hub.update_info('iter', ite + iter_per_epoch * epoch)
