# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import BYOTConnector, ConvModuleConncetor


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
