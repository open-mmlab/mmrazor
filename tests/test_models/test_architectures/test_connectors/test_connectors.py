# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import ConvModuleConncetor, CRDConnector


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
