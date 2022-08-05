# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import ConvModuleConncetor


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
