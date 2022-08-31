# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor import digit_version
from mmrazor.models import (ABLoss, ActivationLoss, ATLoss, DKDLoss, FBKDLoss,
                            FTLoss, InformationEntropyLoss, KDSoftCELoss,
                            OFDLoss, OnehotLikeLoss)


class TestLosses(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feats_1d = torch.randn(5, 6)
        cls.feats_2d = torch.randn(5, 2, 3)
        cls.feats_3d = torch.randn(5, 2, 3, 3)

        num_classes = 6
        cls.labels = torch.randint(0, num_classes, [5])

    def test_ofd_loss(self):
        ofd_loss = OFDLoss()
        self.normal_test_1d(ofd_loss)
        self.normal_test_3d(ofd_loss)

        # test the calculation
        s_feat_0 = torch.Tensor([[1, 1], [2, 2], [3, 3]])
        t_feat_0 = torch.Tensor([[0, 0], [1, 1], [2, 2]])
        ofd_loss_num_0 = ofd_loss.forward(s_feat_0, t_feat_0)
        assert ofd_loss_num_0 != torch.tensor(0.0)

        s_feat_1 = torch.Tensor([[1, 1], [2, 2], [3, 3]])
        t_feat_1 = torch.Tensor([[2, 2], [3, 3], [4, 4]])
        ofd_loss_num_1 = ofd_loss.forward(s_feat_1, t_feat_1)
        assert ofd_loss_num_1 != torch.tensor(0.0)

        s_feat_2 = torch.Tensor([[-3, -3], [-2, -2], [-1, -1]])
        t_feat_2 = torch.Tensor([[-2, -2], [-1, -1], [0, 0]])
        ofd_loss_num_2 = ofd_loss.forward(s_feat_2, t_feat_2)
        assert ofd_loss_num_2 == torch.tensor(0.0)

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

    def test_ft_loss(self):
        ft_loss_cfg = dict(loss_weight=1.0)
        ft_loss = FTLoss(**ft_loss_cfg)

        assert ft_loss.loss_weight == 1.0

        self.normal_test_1d(ft_loss)
        self.normal_test_2d(ft_loss)
        self.normal_test_3d(ft_loss)

    def test_dafl_loss(self):
        dafl_loss_cfg = dict(loss_weight=1.0)
        ac_loss = ActivationLoss(**dafl_loss_cfg, norm_type='abs')
        oh_loss = OnehotLikeLoss(**dafl_loss_cfg)
        ie_loss = InformationEntropyLoss(**dafl_loss_cfg, gather=False)

        # normal test with only one input
        loss_ac = ac_loss.forward(self.feats_1d)
        self.assertTrue(loss_ac.numel() == 1)
        loss_oh = oh_loss.forward(self.feats_1d)
        self.assertTrue(loss_oh.numel() == 1)
        loss_ie = ie_loss.forward(self.feats_1d)
        self.assertTrue(loss_ie.numel() == 1)

        with self.assertRaisesRegex(AssertionError,
                                    '"norm_type" must be "norm" or "abs"'):
            _ = ActivationLoss(**dafl_loss_cfg, norm_type='random')

        # test gather_tensors
        ie_loss = InformationEntropyLoss(**dafl_loss_cfg, gather=True)
        ie_loss.world_size = 2

        if digit_version(torch.__version__) >= digit_version('1.8.0'):
            with self.assertRaisesRegex(
                    RuntimeError,
                    'Default process group has not been initialized'):
                loss_ie = ie_loss.forward(self.feats_1d)
        else:
            with self.assertRaisesRegex(
                    AssertionError,
                    'Default process group is not initialized'):
                loss_ie = ie_loss.forward(self.feats_1d)

    def test_kdSoftce_loss(self):
        kdSoftce_loss_cfg = dict(loss_weight=1.0)
        kdSoftce_loss = KDSoftCELoss(**kdSoftce_loss_cfg)
        # kd soft ce loss requires label logits
        self.normal_test_1d(kdSoftce_loss, labels=True)

    def test_at_loss(self):
        at_loss_cfg = dict(loss_weight=1.0)
        at_loss = ATLoss(**at_loss_cfg)

        assert at_loss.loss_weight == 1.0

        self.normal_test_1d(at_loss)
        self.normal_test_2d(at_loss)
        self.normal_test_3d(at_loss)

    def test_fbkdloss(self):
        fbkdloss_cfg = dict(loss_weight=1.0)
        fbkdloss = FBKDLoss(**fbkdloss_cfg)

        spatial_mask = torch.randn(1, 1, 3, 3)
        channel_mask = torch.randn(1, 4, 1, 1)
        channel_pool_adapt = torch.randn(1, 4)
        relation_adpt = torch.randn(1, 4, 3, 3)

        s_input = (spatial_mask, channel_mask, channel_pool_adapt,
                   spatial_mask, channel_mask, relation_adpt)
        t_input = (spatial_mask, channel_mask, spatial_mask, channel_mask,
                   relation_adpt)

        fbkd_loss = fbkdloss(s_input, t_input)
        self.assertTrue(fbkd_loss.numel() == 1)
