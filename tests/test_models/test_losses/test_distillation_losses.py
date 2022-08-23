# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor import digit_version
from mmrazor.models import (ABLoss, ActivationLoss, DKDLoss,
                            InformationEntropyLoss, KDSoftCELoss,
                            OnehotLikeLoss)


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

        # TODO: configure circle CI to test UT under multi torch versions.
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
