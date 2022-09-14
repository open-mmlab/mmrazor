# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch import nn

from mmrazor.models.task_modules import ParameterRecorder


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.toy_conv = nn.Conv2d(1, 1, 1)
        self.no_record_conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.toy_conv(x)


class TestParameterRecorder(TestCase):

    def test_prepare_from_model(self):

        model = ToyModel()
        recorder = ParameterRecorder('AAA')
        with self.assertRaisesRegex(AssertionError, '"AAA" is not in the'):
            recorder.initialize(model)

        recorder = ParameterRecorder('toy_conv.bias')
        with self.assertRaisesRegex(AssertionError, 'model can not be None'):
            recorder.prepare_from_model()

        recorder.initialize(model)
        bias_weight = recorder.get_record_data()

        self.assertEquals(bias_weight, model.toy_conv.bias)

        with recorder:
            _ = model(torch.randn(1, 1, 1, 1))

        self.assertEquals(bias_weight, model.toy_conv.bias)
