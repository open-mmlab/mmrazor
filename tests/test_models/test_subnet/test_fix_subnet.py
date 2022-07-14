# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch.nn as nn

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.models.mutables import OneShotMutableOP
from mmrazor.models.subnet import (FIX_MUTABLE, export_fix_mutable,
                                   load_fix_subnet)
from mmrazor.registry import MODELS

MODELS.register_module()


class MockModel(nn.Module):

    def __init__(self):
        super().__init__()
        convs1 = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 8, 1),
            'conv2': nn.Conv2d(3, 8, 1),
            'conv3': nn.Conv2d(3, 8, 1),
        })
        convs2 = nn.ModuleDict({
            'conv1': nn.Conv2d(8, 16, 1),
            'conv2': nn.Conv2d(8, 16, 1),
            'conv3': nn.Conv2d(8, 16, 1),
        })

        self.mutable1 = OneShotMutableOP(convs1)
        self.mutable2 = OneShotMutableOP(convs2)

    def forward(self, x):
        x = self.mutable1(x)
        x = self.mutable2(x)
        return x


class TestFixSubnet(TestCase):

    def test_load_fix_subnet(self):
        # fix subnet is str
        fix_subnet = 'tests/data/test_models/test_subnet/mockmodel_subnet.yaml'  # noqa: E501
        model = MockModel()

        load_fix_subnet(model, fix_subnet)

        # fix subnet is dict
        fix_subnet = {
            'mutable1': 'conv1',
            'mutable2': 'conv2',
        }

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        with pytest.raises(TypeError):
            # type int is not supported.
            model = MockModel()
            load_fix_subnet(model, fix_subnet=10)

    def test_export_fix_subnet(self):
        # get FixSubnet
        fix_subnet = {
            'mutable1': 'conv1',
            'mutable2': 'conv2',
        }

        model = MockModel()
        load_fix_subnet(model, fix_subnet)

        with pytest.raises(AssertionError):
            exported_fix_subnet: FIX_MUTABLE = export_fix_mutable(model)

        model = MockModel()
        model.mutable1.current_choice = 'conv1'
        model.mutable2.current_choice = 'conv2'
        exported_fix_subnet = export_fix_mutable(model)

        self.assertDictEqual(fix_subnet, exported_fix_subnet)
