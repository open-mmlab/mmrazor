# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import BaseDataElement

from mmrazor.models import ABLoss, CRDLoss, DKDLoss


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

    def _mock_crd_data_sample(self, sample_idx_list):
        data_samples = []
        for _idx in sample_idx_list:
            data_sample = BaseDataElement()
            data_sample.set_data(dict(sample_idx=_idx))
            data_samples.append(data_sample)
        return data_samples

    def test_crd_loss(self):
        crd_loss = CRDLoss(**dict(neg_num=5, sample_n=10, dim_out=6))
        sample_idx_list = torch.tensor(list(range(5)))
        data_samples = self._mock_crd_data_sample(sample_idx_list)
        loss = crd_loss.forward(self.feats_1d, self.feats_1d, data_samples)
        self.assertTrue(loss.numel() == 1)

        # test the calculation
        s_feat_0 = torch.randn((5, 6))
        t_feat_0 = torch.randn((5, 6))
        crd_loss_num_0 = crd_loss.forward(s_feat_0, t_feat_0, data_samples)
        assert crd_loss_num_0 != torch.tensor(0.0)

        s_feat_1 = torch.randn((5, 6))
        t_feat_1 = torch.rand((5, 6))
        sample_idx_list_1 = torch.tensor(list(range(5)))
        data_samples_1 = self._mock_crd_data_sample(sample_idx_list_1)
        crd_loss_num_1 = crd_loss.forward(s_feat_1, t_feat_1, data_samples_1)
        assert crd_loss_num_1 != torch.tensor(0.0)

    def test_dkd_loss(self):
        dkd_loss_cfg = dict(loss_weight=1.0)
        dkd_loss = DKDLoss(**dkd_loss_cfg)
        # dkd requires label logits
        self.normal_test_1d(dkd_loss, labels=True)
