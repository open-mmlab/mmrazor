# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import ABLoss, DKDLoss


class TestLosses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feats_1d = torch.randn(5, 6)
        cls.feats_2d = torch.randn(5, 2, 3)
        cls.feats_3d = torch.randn(5, 2, 3, 3)

        num_classes = 6
        cls.labels = torch.randint(0, num_classes, [5])

    def normal_test_1d(self, loss_instance, labels=False):
        args = tuple([self.feats_1d, self.feats_1d])
        if labels:
            args += (self.labels, )
        loss_1d = loss_instance.forward(*args)
        self.assertTrue(loss_1d.numel() == 1)

    def normal_test_2d(self, loss_instance, labels=False):
        args = tuple([self.feats_2d, self.feats_2d])
        if labels:
            args += (self.labels, )
        loss_2d = loss_instance.forward(*args)
        self.assertTrue(loss_2d.numel() == 1)

    def normal_test_3d(self, loss_instance, labels=False):
        args = tuple([self.feats_3d, self.feats_3d])
        if labels:
            args += (self.labels, )
        loss_3d = loss_instance.forward(*args)
        self.assertTrue(loss_3d.numel() == 1)

    def test_ab_loss(self):
        ab_loss_cfg = dict(loss_weight=1.0, margin=1.0)
        ab_loss = ABLoss(**ab_loss_cfg)
        self.normal_test_1d(ab_loss)
        self.normal_test_2d(ab_loss)
        self.normal_test_3d(ab_loss)

    def test_dkd_loss(self):
        dkd_loss_cfg = dict(loss_weight=1.0)
        dkd_loss = DKDLoss(**dkd_loss_cfg)
        # dkd requires label logits
        self.normal_test_1d(dkd_loss, labels=True)
