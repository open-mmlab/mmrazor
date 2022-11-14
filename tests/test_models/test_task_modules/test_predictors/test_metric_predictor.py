# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase

import numpy as np
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models import OneShotMutableOP
from mmrazor.registry import TASK_UTILS

convs = nn.ModuleDict({
    'conv1': nn.Conv2d(3, 8, 1),
    'conv2': nn.Conv2d(3, 8, 1),
    'conv3': nn.Conv2d(3, 8, 1),
})
MutableOP = OneShotMutableOP(convs)


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.mutable = MutableOP
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


class TestMetricPredictorWithGP(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.search_groups = {0: [MutableOP], 1: [MutableOP]}
        self.candidates = [{0: 'conv1'}, {0: 'conv2'}, {0: 'conv3'}]
        predictor_cfg = dict(
            type='MetricPredictor',
            handler_cfg=dict(type='GaussProcessHandler'),
            search_groups=self.search_groups,
            train_samples=4,
        )
        self.predictor = TASK_UTILS.build(predictor_cfg)
        self.model = ToyModel()

    def generate_data(self):
        inputs = []
        for candidate in self.candidates:
            inputs.append(self.predictor.model2vector(candidate))
        inputs = np.array(inputs)
        labels = np.random.rand(3)
        return inputs, labels

    def test_init_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.assertFalse(self.predictor.initialize)
        self.predictor.fit(inputs, labels)
        self.assertTrue(self.predictor.initialize)

    def test_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.predictor.fit(inputs, labels)

        metrics = self.predictor.predict(self.model)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['accuracy_top-1'], 0.0)


class TestMetricPredictorWithCart(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.search_groups = {0: [MutableOP], 1: [MutableOP]}
        self.candidates = [{0: 'conv1'}, {0: 'conv2'}, {0: 'conv3'}]
        predictor_cfg = dict(
            type='MetricPredictor',
            handler_cfg=dict(type='CartsHandler'),
            search_groups=self.search_groups,
            train_samples=4,
        )
        self.predictor = TASK_UTILS.build(predictor_cfg)
        self.model = ToyModel()

    def generate_data(self):
        inputs = []
        for candidate in self.candidates:
            inputs.append(self.predictor.model2vector(candidate))
        inputs = np.array(inputs)
        labels = np.random.rand(3)
        return inputs, labels

    def test_init_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.assertFalse(self.predictor.initialize)
        self.predictor.fit(inputs, labels)
        self.assertTrue(self.predictor.initialize)

    def test_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.predictor.fit(inputs, labels)

        metrics = self.predictor.predict(self.model)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['accuracy_top-1'], 0.0)


class TestMetricPredictorWithRBF(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.search_groups = {0: [MutableOP], 1: [MutableOP]}
        self.candidates = [{0: 'conv1'}, {0: 'conv2'}, {0: 'conv3'}]
        predictor_cfg = dict(
            type='MetricPredictor',
            handler_cfg=dict(type='RBFHandler'),
            search_groups=self.search_groups,
            train_samples=4,
        )
        self.predictor = TASK_UTILS.build(predictor_cfg)
        self.model = ToyModel()

    def generate_data(self):
        inputs = []
        for candidate in self.candidates:
            inputs.append(self.predictor.model2vector(candidate))
        inputs = np.array(inputs)
        labels = np.random.rand(3)
        return inputs, labels

    def test_init_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.assertFalse(self.predictor.initialize)
        self.predictor.fit(inputs, labels)
        self.assertTrue(self.predictor.initialize)

    def test_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.predictor.fit(inputs, labels)

        metrics = self.predictor.predict(self.model)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['accuracy_top-1'], 0.0)


class TestMetricPredictorWithMLP(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.search_groups = {0: [MutableOP], 1: [MutableOP]}
        self.candidates = [{0: 'conv1'}, {0: 'conv2'}, {0: 'conv3'}]
        predictor_cfg = dict(
            type='MetricPredictor',
            handler_cfg=dict(type='MLPHandler'),
            search_groups=self.search_groups,
            train_samples=4,
        )
        self.predictor = TASK_UTILS.build(predictor_cfg)
        self.model = ToyModel()

    def generate_data(self):
        inputs = []
        for candidate in self.candidates:
            inputs.append(self.predictor.model2vector(candidate))
        inputs = np.array(inputs)
        labels = np.random.rand(3)
        return inputs, labels

    def test_init_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.assertFalse(self.predictor.initialize)
        self.predictor.fit(inputs, labels)
        self.assertTrue(self.predictor.initialize)

    def test_predictor(self):
        self.model.mutable.current_choice = 'conv1'
        inputs, labels = self.generate_data()
        self.predictor.fit(inputs, labels)

        metrics = self.predictor.predict(self.model)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['accuracy_top-1'], 0.0)
