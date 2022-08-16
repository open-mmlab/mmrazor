# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import ABLoss, CRDLoss


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

    def test_ab_loss(self):
        ab_loss_cfg = dict(loss_weight=1.0, margin=1.0)
        ab_loss = ABLoss(**ab_loss_cfg)
        self.normal_test_1d(ab_loss)
        self.normal_test_2d(ab_loss)
        self.normal_test_3d(ab_loss)

    def test_crd_loss(self):
        crd_loss = CRDLoss(dict(neg_num=5, sample_n=10, dim_out=6))
        idx = torch.tensor(list(range(5)))
        loss = crd_loss.forward(self.feats_1d, self.feats_1d, idx=idx)
        self.assertTrue(loss.numel() == 1)

        # test the calculation
        s_feat_0 = torch.randn((5, 6))
        t_feat_0 = torch.randn((5, 6))
        crd_loss_num_0 = crd_loss.forward(s_feat_0, t_feat_0, idx=idx)
        assert crd_loss_num_0 != torch.tensor(0.0)

        s_feat_1 = torch.randn((5, 6))
        t_feat_1 = torch.rand((5, 6))
        idx_1 = torch.tensor(list(range(5)))
        crd_loss_num_1 = crd_loss.forward(s_feat_1, t_feat_1, idx=idx_1)
        assert crd_loss_num_1 != torch.tensor(0.0)
