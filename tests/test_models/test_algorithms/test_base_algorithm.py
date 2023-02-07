# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor, BaseModel

from mmrazor.models import BaseAlgorithm
from mmrazor.models.task_modules import ModuleOutputsRecorder
from mmrazor.registry import MODELS


@MODELS.register_module()
class CustomDataPreprocessor(BaseDataPreprocessor):

    def forward(self, data, training=False):
        if training:
            return 1
        else:
            return 2


@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.relu(self.bn(self.conv(batch_inputs)))
            return dict(loss=out)
        elif mode == 'predict':
            out = self.relu(self.bn(self.conv(batch_inputs) + 1))
            return out
        elif mode == 'tensor':
            out = self.relu(self.bn(self.conv(batch_inputs) + 2))
            return out


class TestBaseAlgorithm(TestCase):

    def test_init(self):
        # initiate model without `data_preprocessor`
        model = ToyModel()
        alg = BaseAlgorithm(ToyModel())
        self.assertIsInstance(alg.data_preprocessor, BaseDataPreprocessor)
        # self.assertIs(alg.data_preprocessor, model.data_preprocessor)

        # initiate model with unbuilt `data_preprocessor`.
        data_preprocessor = dict(type='mmrazor.CustomDataPreprocessor')
        alg = BaseAlgorithm(ToyModel(), data_preprocessor=data_preprocessor)
        self.assertIsInstance(alg.data_preprocessor, CustomDataPreprocessor)

        # initiate algorithm with built `data_preprocessor`.
        data_preprocessor = CustomDataPreprocessor()
        alg = BaseAlgorithm(
            ToyModel(data_preprocessor), data_preprocessor=data_preprocessor)
        self.assertIs(alg.data_preprocessor, data_preprocessor)
        self.assertIs(alg.data_preprocessor,
                      alg.architecture.data_preprocessor)
        alg = BaseAlgorithm(
            ToyModel(data_preprocessor), data_preprocessor=None)
        self.assertIs(alg.data_preprocessor, data_preprocessor)
        self.assertIs(alg.data_preprocessor,
                      alg.architecture.data_preprocessor)

        # initiate algorithm with built `model`.
        model = ToyModel()
        alg = BaseAlgorithm(model)
        self.assertIs(alg.architecture, model)

        # initiate algorithm with unbuilt `model`.
        model = dict(type='ToyModel')
        alg = BaseAlgorithm(model)
        self.assertIsInstance(alg.architecture, ToyModel)

        # initiate algorithm with error type `model`.
        with self.assertRaisesRegex(TypeError, 'architecture should be'):
            BaseAlgorithm(architecture=[model])

    def test_forward(self):

        model = ToyModel()
        alg = BaseAlgorithm(model)

        inputs = torch.randn(1, 3, 8, 8)

        loss = alg(inputs, mode='loss')
        loss_ = alg.loss(inputs)
        self.assertEqual(loss['loss'].sum(), loss_['loss'].sum())

        predict = alg(inputs, mode='predict')
        predict_ = alg._predict(inputs)
        self.assertEqual(predict.sum(), predict_.sum())

        tensor = alg(inputs, mode='tensor')
        tensor_ = alg._forward(inputs)
        self.assertEqual(tensor.sum(), tensor_.sum())

        with self.assertRaisesRegex(RuntimeError, 'Invalid mode "A"'):
            alg(inputs, mode='A')

    def test_set_module_inplace_false(self):
        inputs = torch.randn(1, 3, 8, 8)

        model = ToyModel()
        res_before = model(inputs)
        _ = BaseAlgorithm(model)

        r1 = ModuleOutputsRecorder('bn')
        r1.initialize(model)
        with r1:
            res_after = model(inputs)
        self.assertIs(torch.equal(res_before, res_after), True)

        self.assertIs(model.relu.inplace, False)

        self.assertIs(
            torch.equal(r1.data_buffer[0], model.bn(model.conv(inputs) + 2)),
            True)
