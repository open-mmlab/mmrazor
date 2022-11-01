# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models import SPOS, OneShotModuleMutator, OneShotMutableOP
from mmrazor.registry import MODELS


@MODELS.register_module()
class ToySearchableModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        convs = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 8, 1),
            'conv2': nn.Conv2d(3, 8, 1),
            'conv3': nn.Conv2d(3, 8, 1),
        })
        self.mutable = OneShotMutableOP(convs)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.bn(self.mutable(batch_inputs))
            return dict(loss=out)
        elif mode == 'predict':
            out = self.bn(self.mutable(batch_inputs)) + 1
            return out
        elif mode == 'tensor':
            out = self.bn(self.mutable(batch_inputs)) + 2
            return out


class TestSPOS(TestCase):

    def test_init(self):
        # initiate spos when `norm_training` is True.
        model = ToySearchableModel()
        mutator = OneShotModuleMutator()
        alg = SPOS(model, mutator, norm_training=True)
        alg.eval()
        self.assertTrue(model.bn.training)

        # initiate spos with built `mutator`.
        model = ToySearchableModel()
        mutator = OneShotModuleMutator()
        alg = SPOS(model, mutator)
        self.assertIs(alg.mutator, mutator)

        # initiate spos with unbuilt `mutator`.
        mutator = dict(type='OneShotModuleMutator')
        alg = SPOS(model, mutator)
        self.assertIsInstance(alg.mutator, OneShotModuleMutator)

        # initiate spos when `fix_subnet` is not None.
        fix_subnet = {'mutable': {'chosen': 'conv1'}}
        alg = SPOS(model, mutator, fix_subnet=fix_subnet)
        self.assertEqual(alg.architecture.mutable.num_choices, 1)

        # initiate spos with error type `mutator`.
        with self.assertRaisesRegex(TypeError, 'mutator should be'):
            SPOS(model, model)

    def test_forward_loss(self):
        inputs = torch.randn(1, 3, 8, 8)
        model = ToySearchableModel()

        # supernet
        mutator = OneShotModuleMutator()
        alg = SPOS(model, mutator)
        loss = alg(inputs, mode='loss')
        self.assertIsInstance(loss, dict)

        # subnet
        fix_subnet = {'mutable': {'chosen': 'conv1'}}
        alg = SPOS(model, fix_subnet=fix_subnet)
        loss = alg(inputs, mode='loss')
        self.assertIsInstance(loss, dict)
