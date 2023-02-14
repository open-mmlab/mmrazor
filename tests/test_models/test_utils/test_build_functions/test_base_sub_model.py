# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmengine import fileio
from mmengine.model import BaseModel

from mmrazor.registry import MODELS


class TestBaseSubModel(TestCase):

    def test_build_subnet_prune_from_cfg_by_mutator(self):
        mutator_cfg = fileio.load('tests/data/test_registry/subnet.json')
        init_cfg = dict(
            type='Pretrained',
            checkpoint='tests/data/test_registry/subnet_weight.pth')
        # test fix subnet
        model_cfg = dict(
            # use mmrazor's build_func
            type='mmrazor.sub_model',
            cfg=dict(
                cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py',
                pretrained=False),
            fix_subnet=mutator_cfg,
            mode='mutator',
            init_cfg=init_cfg)
        model = MODELS.build(model_cfg)
        self.assertTrue(isinstance(model, BaseModel))

    def test_build_subnet_prune_from_cfg_by_mutable(self):
        mutator_cfg = fileio.load('tests/data/test_registry/subnet.json')
        init_cfg = dict(
            type='Pretrained',
            checkpoint='tests/data/test_registry/subnet_weight.pth')
        # test fix subnet
        model_cfg = dict(
            # use mmrazor's build_func
            type='mmrazor.sub_model',
            cfg=dict(
                cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py',
                pretrained=False),
            fix_subnet=mutator_cfg,
            mode='mutable',
            init_cfg=init_cfg)
        model = MODELS.build(model_cfg)
        self.assertTrue(isinstance(model, BaseModel))
