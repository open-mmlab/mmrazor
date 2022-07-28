# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import L2Loss


class TestLosses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feats_1d = torch.randn(5, 6)
        cls.feats_3d = torch.randn(5, 2, 3, 3)

    def normal_test_1d(self, loss_instance):
        loss_1d = loss_instance.forward(self.feats_1d, self.feats_1d)
        self.assertTrue(loss_1d.numel() == 1)

    def normal_test_3d(self, loss_instance):
        loss_3d = loss_instance.forward(self.feats_3d, self.feats_3d)
        self.assertTrue(loss_3d.numel() == 1)

    def test_l2_loss(self):
        l2_loss_cfg = dict(loss_weight=10)
        l2_loss = L2Loss(**l2_loss_cfg)
        self.normal_test_1d(l2_loss)
        self.normal_test_3d(l2_loss)
