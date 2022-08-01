# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import ConvConnector


class TestConnector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s_feat = torch.randn(1, 1, 5, 5)
        cls.t_feat = torch.randn(1, 3, 5, 5)

    def test_conv_connector(self):
        conv_connector_cfg = dict(in_channel=1, out_channel=3, use_relu=True)
        conv_connector = ConvConnector(**conv_connector_cfg)
        conv_connector.init_weights()

        output = conv_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()

        conv_connector_cfg['use_norm'] = True
        with self.assertRaisesRegex(
            AssertionError, '"use_norm" is True but "norm_cfg is None."'):
            _ = ConvConnector(**conv_connector_cfg)

        conv_connector_cfg['norm_cfg'] = 'BN'
        with self.assertRaisesRegex(TypeError, 'cfg must be a dict'):
            _ = ConvConnector(**conv_connector_cfg)
