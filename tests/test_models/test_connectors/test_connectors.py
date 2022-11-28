# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor.models.architectures.connectors import ConvModuleConnector


class TestConnector(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s_feat = torch.randn(1, 1, 5, 5)
        cls.t_feat = torch.randn(1, 3, 5, 5)

    def test_convmodule_connector(self):
        convmodule_connector_cfg = dict(in_channel=1, out_channel=3)
        relu_connector = ConvModuleConnector(**convmodule_connector_cfg)

        output = relu_connector.forward_train(self.s_feat)
        assert output.size() == self.t_feat.size()
