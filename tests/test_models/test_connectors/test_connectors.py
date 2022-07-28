# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models import BNConnector, ReLUConnector, SingleConvConnector


class TestConnector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s_feat = torch.randn(1, 1, 5, 5)
        cls.t_feat = torch.randn(1, 3, 5, 5)

    def test_singleconv_connector(self):
        singleconv_connector_cfg = dict(in_channel=1, out_channel=3)
        singleconv_connector = SingleConvConnector(**singleconv_connector_cfg)

        output = singleconv_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()

    def test_bn_connector(self):
        bn_connector_cfg = dict(in_channel=1, out_channel=3)
        bn_connector = BNConnector(**bn_connector_cfg)

        output = bn_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()

    def test_relu_connector(self):
        relu_connector_cfg = dict(in_channel=1, out_channel=3)
        relu_connector = ReLUConnector(**relu_connector_cfg)

        output = relu_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()
