# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from typing import Dict, Optional, Union
from unittest import TestCase

import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModel

from mmrazor.models import *  # noqa: F403, F401
from mmrazor.models.algorithms.base import BaseAlgorithm
from mmrazor.models.mutables import OneShotMutableOP
from mmrazor.registry import MODELS
from mmrazor.structures import load_fix_subnet
from mmrazor.utils import ValidFixMutable


@MODELS.register_module()
class MockModel(BaseModel):

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


@MODELS.register_module()
class MockAlgorithm(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 fix_subnet: Optional[ValidFixMutable] = None):
        super().__init__(architecture)

        if fix_subnet is not None:
            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self, fix_subnet, prefix='architecture.')
            self.is_supernet = False
        else:
            self.is_supernet = True


class TestRegistry(TestCase):

    def setUp(self) -> None:
        self.arch_cfg_path = dict(
            cfg_path='mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            pretrained=False)

        return super().setUp()

    def test_build_razor_from_cfg(self):
        # test cfg_path
        # TODO relay on mmengine:HAOCHENYE/config_new_feature
        # model = MODELS.build(self.arch_cfg_path)
        # self.assertIsNotNone(model)

        # test fix subnet
        cfg = Config.fromfile(
            'tests/data/test_registry/registry_subnet_config.py')
        model = MODELS.build(cfg.model)

        # test return architecture
        cfg = Config.fromfile(
            'tests/data/test_registry/registry_architecture_config.py')
        model = MODELS.build(cfg.model)
        self.assertTrue(isinstance(model, BaseModel))


if __name__ == '__main__':
    unittest.main()
