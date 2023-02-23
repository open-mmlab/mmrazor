# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union
from unittest import TestCase

import torch
from mmdet.models.utils import unpack_gt_instances
from mmdet.testing import demo_mm_inputs
from mmengine.structures import BaseDataElement
from torch import Tensor

from mmrazor import digit_version
from mmrazor.models import (ABLoss, ActivationLoss, ATLoss, CRDLoss, DKDLoss,
                            FBKDLoss, FGDLoss, FTLoss, InformationEntropyLoss,
                            KDSoftCELoss, MGDLoss, OFDLoss, OnehotLikeLoss,
                            PKDLoss)


# copied from mmyolo
def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:
    """Split batch_gt_instances with batch size, from [all_gt_bboxes, 6] to.

    [batch_size, number_gt, 5]. If some shape of single batch smaller than
    gt bbox len, then using [-1., 0., 0., 0., 0.] to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, 6]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape [batch_size, number_gt, 5]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances])
        # fill [0., 0., 0., 0., 0.] if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            batch_instance_list.append(
                torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], 5], 0)
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0)

        return torch.stack(batch_instance_list)
    else:
        # faster version
        # format of batch_gt_instances:
        # [img_ind, cls_ind, x1, y1, x2, y2]

        # sqlit batch gt instance [all_gt_bboxes, 6] ->
        # [batch_size, max_gt_bbox_len, 5]
        assert isinstance(batch_gt_instances, Tensor)
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(
                return_counts=True)[1].max()
            # fill [0., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance = torch.zeros((batch_size, max_gt_bbox_len, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[
                        match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

        return batch_instance


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

    def test_pkdloss(self):
        pkd_loss = PKDLoss(loss_weight=1.0)
        feats_S, feats_T = torch.rand(2, 256, 4, 4), torch.rand(2, 256, 4, 4)
        loss = pkd_loss(feats_S, feats_T)
        self.assertTrue(loss.numel() == 1)
        self.assertTrue(0. <= loss <= 1.)

        num_stages = 4
        feats_S = (torch.rand(2, 256, 4, 4) for _ in range(num_stages))
        feats_T = (torch.rand(2, 256, 4, 4) for _ in range(num_stages))
        loss = pkd_loss(feats_S, feats_T)
        self.assertTrue(loss.numel() == 1)
        self.assertTrue(0. <= loss <= num_stages * 1.)

        feats_S, feats_T = torch.rand(2, 256, 2, 2), torch.rand(2, 256, 4, 4)
        loss = pkd_loss(feats_S, feats_T)
        self.assertTrue(loss.numel() == 1)
        self.assertTrue(0. <= loss <= 1.)

        pkd_loss = PKDLoss(loss_weight=1.0, resize_stu=False)
        feats_S, feats_T = torch.rand(2, 256, 2, 2), torch.rand(2, 256, 4, 4)
        loss = pkd_loss(feats_S, feats_T)
        self.assertTrue(loss.numel() == 1)
        self.assertTrue(0. <= loss <= 1.)

    def test_mgd_loss(self):
        mgd_loss = MGDLoss(alpha_mgd=0.00002)
        feats_S, feats_T = torch.rand(2, 256, 4, 4), torch.rand(2, 256, 4, 4)
        loss = mgd_loss(feats_S, feats_T)
        self.assertTrue(loss.numel() == 1)

    def test_fgd_loss(self):
        fgd_loss = FGDLoss(in_channels=3)
        packed_inputs = demo_mm_inputs(2, [[3, 320, 128], [3, 125, 320]])
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas =\
            unpack_gt_instances(packed_inputs['data_samples'])
        gt_info = gt_instances_preprocess(batch_gt_instances, 2)
        for meta in batch_img_metas:
            meta.update({'batch_input_shape': meta['img_shape']})

        preds_S = torch.rand(2, 3, 80, 32)
        preds_T = torch.rand(2, 3, 80, 32)

        loss = fgd_loss(preds_S, preds_T, gt_info, batch_img_metas)
        self.assertTrue(loss.numel() == 1)
