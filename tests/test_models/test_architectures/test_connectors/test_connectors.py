# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import (BYOTConnector, ConvModuleConncetor,
                            FBKDStudentConnector, FBKDTeacherConnector,
                            Paraphraser, Translator)


class TestConnector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s_feat = torch.randn(1, 1, 5, 5)
        cls.t_feat = torch.randn(1, 3, 5, 5)

    def test_convmodule_connector(self):
        convmodule_connector_cfg = dict(
            in_channel=1, out_channel=3, norm_cfg=dict(type='BN'))
        convmodule_connector = ConvModuleConncetor(**convmodule_connector_cfg)

        output = convmodule_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()

        convmodule_connector_cfg['order'] = ('conv', 'norm')
        with self.assertRaises(AssertionError):
            _ = ConvModuleConncetor(**convmodule_connector_cfg)

        convmodule_connector_cfg['act_cfg'] = 'ReLU'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConncetor(**convmodule_connector_cfg)

        convmodule_connector_cfg['norm_cfg'] = 'BN'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConncetor(**convmodule_connector_cfg)

        convmodule_connector_cfg['conv_cfg'] = 'conv2d'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConncetor(**convmodule_connector_cfg)

    def test_ft_connector(self):
        stu_connector = Translator(**dict(in_channel=1, out_channel=2))

        tea_connector = Paraphraser(**dict(in_channel=3, out_channel=2))

        s_connect = stu_connector.forward_train(self.s_feat)
        t_connect = tea_connector.forward_train(self.t_feat)
        assert s_connect.size() == t_connect.size()
        t_pretrain = tea_connector.forward_pretrain(self.t_feat)
        assert t_pretrain.size() == torch.Size([1, 3, 5, 5])

    def test_byot_connector(self):
        byot_connector_cfg = dict(
            in_channel=16,
            out_channel=32,
            num_classes=10,
            expansion=4,
            pool_size=4,
            kernel_size=3,
            stride=2,
            init_cfg=None)
        byot_connector = BYOTConnector(**byot_connector_cfg)

        s_feat = torch.randn(1, 16 * 4, 8, 8)
        t_feat = torch.randn(1, 32 * 4)
        labels = torch.randn(1, 10)

        output, logits = byot_connector.forward_train(s_feat)
        assert output.size() == t_feat.size()
        assert logits.size() == labels.size()

    def test_fbkd_connector(self):
        fbkd_stuconnector_cfg = dict(
            in_channel=16, inter_channel=8, downsample_stride=2)
        fbkd_stuconnector = FBKDStudentConnector(**fbkd_stuconnector_cfg)

        fbkd_teaconnector_cfg = dict(
            in_channel=16, inter_channel=8, downsample_stride=2)
        fbkd_teaconnector = FBKDTeacherConnector(**fbkd_teaconnector_cfg)

        s_feat = torch.randn(1, 16, 8, 8)
        t_feat = torch.randn(1, 16, 8, 8)

        s_output = fbkd_stuconnector(s_feat)
        t_output = fbkd_teaconnector(t_feat)

        assert len(s_output) == 6
        assert len(t_output) == 5
        assert torch.equal(t_output[-1], t_feat)
