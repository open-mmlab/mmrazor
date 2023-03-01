# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import (BYOTConnector, ConvModuleConnector, CRDConnector,
                            FBKDStudentConnector, FBKDTeacherConnector,
                            MGDConnector, NormConnector, Paraphraser,
                            TorchFunctionalConnector, TorchNNConnector,
                            Translator)


class TestConnector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s_feat = torch.randn(1, 1, 5, 5)
        cls.t_feat = torch.randn(1, 3, 5, 5)

    def test_convmodule_connector(self):
        convmodule_connector_cfg = dict(
            in_channel=1, out_channel=3, norm_cfg=dict(type='BN'))
        convmodule_connector = ConvModuleConnector(**convmodule_connector_cfg)

        output = convmodule_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()

        convmodule_connector_cfg['order'] = ('conv', 'norm')
        with self.assertRaises(AssertionError):
            _ = ConvModuleConnector(**convmodule_connector_cfg)

        convmodule_connector_cfg['act_cfg'] = 'ReLU'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConnector(**convmodule_connector_cfg)

        convmodule_connector_cfg['norm_cfg'] = 'BN'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConnector(**convmodule_connector_cfg)

        convmodule_connector_cfg['conv_cfg'] = 'conv2d'
        with self.assertRaises(AssertionError):
            _ = ConvModuleConnector(**convmodule_connector_cfg)

    def test_crd_connector(self):
        dim_out = 128
        crd_stu_connector = CRDConnector(
            **dict(dim_in=1 * 5 * 5, dim_out=dim_out))

        crd_tea_connector = CRDConnector(
            **dict(dim_in=3 * 5 * 5, dim_out=dim_out))

        assert crd_stu_connector.linear.in_features == 1 * 5 * 5
        assert crd_stu_connector.linear.out_features == dim_out
        assert crd_tea_connector.linear.in_features == 3 * 5 * 5
        assert crd_tea_connector.linear.out_features == dim_out

        s_output = crd_stu_connector.forward_train(self.s_feat)
        t_output = crd_tea_connector.forward_train(self.t_feat)
        assert s_output.size() == t_output.size()

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
            in_channels=16, reduction=2, sub_sample=True)
        fbkd_stuconnector = FBKDStudentConnector(**fbkd_stuconnector_cfg)

        fbkd_teaconnector_cfg = dict(
            in_channels=16, reduction=2, sub_sample=True)
        fbkd_teaconnector = FBKDTeacherConnector(**fbkd_teaconnector_cfg)

        s_feat = torch.randn(1, 16, 8, 8)
        t_feat = torch.randn(1, 16, 8, 8)

        s_output = fbkd_stuconnector(s_feat)
        t_output = fbkd_teaconnector(t_feat)

        assert len(s_output) == 6
        assert len(t_output) == 5
        assert torch.equal(t_output[-1], t_feat)

    def test_torch_connector(self):
        tensor1 = torch.rand(3, 3, 16, 16)
        functional_pool_connector = TorchFunctionalConnector(
            function_name='avg_pool2d', func_args=dict(kernel_size=4))
        tensor2 = functional_pool_connector.forward_train(tensor1)
        assert tensor2.shape == torch.Size([3, 3, 4, 4])

        with self.assertRaises(AssertionError):
            functional_pool_connector = TorchFunctionalConnector()
        with self.assertRaises(ValueError):
            functional_pool_connector = TorchFunctionalConnector(
                function_name='fake')

        nn_pool_connector = TorchNNConnector(
            module_name='AvgPool2d', module_args=dict(kernel_size=4))
        tensor3 = nn_pool_connector.forward_train(tensor1)
        assert tensor3.shape == torch.Size([3, 3, 4, 4])
        assert torch.equal(tensor2, tensor3)

        with self.assertRaises(AssertionError):
            functional_pool_connector = TorchFunctionalConnector()
        with self.assertRaises(ValueError):
            functional_pool_connector = TorchNNConnector(module_name='fake')

    def test_mgd_connector(self):
        s_feat = torch.randn(1, 16, 8, 8)
        mgd_connector1 = MGDConnector(
            student_channels=16, teacher_channels=16, lambda_mgd=0.65)
        mgd_connector2 = MGDConnector(
            student_channels=16, teacher_channels=32, lambda_mgd=0.65)
        s_output1 = mgd_connector1.forward_train(s_feat)
        s_output2 = mgd_connector2.forward_train(s_feat)

        assert s_output1.shape == torch.Size([1, 16, 8, 8])
        assert s_output2.shape == torch.Size([1, 32, 8, 8])

        mgd_connector1 = MGDConnector(
            student_channels=16,
            teacher_channels=16,
            lambda_mgd=0.65,
            mask_on_channel=True)
        mgd_connector2 = MGDConnector(
            student_channels=16,
            teacher_channels=32,
            lambda_mgd=0.65,
            mask_on_channel=True)
        s_output1 = mgd_connector1.forward_train(s_feat)
        s_output2 = mgd_connector2.forward_train(s_feat)

        assert s_output1.shape == torch.Size([1, 16, 8, 8])
        assert s_output2.shape == torch.Size([1, 32, 8, 8])

    def test_norm_connector(self):
        s_feat = torch.randn(2, 3, 2, 2)
        norm_cfg = dict(type='BN', affine=False, track_running_stats=False)
        norm_connector = NormConnector(3, norm_cfg)
        output = norm_connector.forward_train(s_feat)

        assert output.shape == torch.Size([2, 3, 2, 2])
