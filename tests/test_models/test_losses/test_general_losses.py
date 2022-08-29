# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models import L1Loss, L2Loss


class TestLosses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feats_1d = torch.randn(5, 6)
        cls.feats_2d = torch.randn(5, 2, 3)
        cls.feats_3d = torch.randn(5, 2, 3, 3)

    def normal_test_1d(self, loss_instance):
        loss_1d = loss_instance.forward(self.feats_1d, self.feats_1d)
        self.assertTrue(loss_1d.numel() == 1)

    def normal_test_2d(self, loss_instance):
        loss_2d = loss_instance.forward(self.feats_2d, self.feats_2d)
        self.assertTrue(loss_2d.numel() == 1)

    def normal_test_3d(self, loss_instance):
        loss_3d = loss_instance.forward(self.feats_3d, self.feats_3d)
        self.assertTrue(loss_3d.numel() == 1)

    def test_l1_loss(self):
        l1_loss_cfg = dict(loss_weight=10)
        l1_loss = L1Loss(**l1_loss_cfg)
        self.normal_test_1d(l1_loss)
        self.normal_test_2d(l1_loss)
        self.normal_test_3d(l1_loss)

        l1_loss_cfg = dict(loss_weight=10, reduction='avg')
        with pytest.raises(AssertionError):
            l1_loss = L1Loss(**l1_loss_cfg)

    def test_l2_loss(self):
        l2_loss_cfg = dict(loss_weight=10, normalize=True)
        l2_loss = L2Loss(**l2_loss_cfg)
        self.normal_test_1d(l2_loss)
        self.normal_test_2d(l2_loss)
        self.normal_test_3d(l2_loss)

        l2_loss_cfg['div_element'] = True
        l2_loss = L2Loss(**l2_loss_cfg)
        self.normal_test_1d(l2_loss)
        self.normal_test_2d(l2_loss)
        self.normal_test_3d(l2_loss)
