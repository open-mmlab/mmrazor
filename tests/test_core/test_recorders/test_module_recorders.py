# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch import nn

from mmrazor.models.task_modules import (ModuleInputsRecorder,
                                         ModuleOutputsRecorder)


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class TestModuleOutputsRecorder(TestCase):

    def test_prepare_from_model(self):

        recorder = ModuleOutputsRecorder('conv1')
        with self.assertRaisesRegex(AssertionError, 'model can not be'):
            recorder.prepare_from_model()

        recorder = ModuleOutputsRecorder('conv3')
        model = ToyModel()
        with self.assertRaisesRegex(AssertionError, '"conv3" is not in'):
            recorder.prepare_from_model(model)

        recorder = ModuleOutputsRecorder('conv2')
        model = ToyModel()
        recorder.prepare_from_model(model)

    def test_module_outputs(self):

        recorder = ModuleOutputsRecorder('conv2')
        model = ToyModel()
        recorder.initialize(model)

        with recorder:
            self.assertTrue(recorder.recording)
            res = model(torch.randn(1, 1, 1, 1))

        self.assertEquals(res, recorder.get_record_data())

        with recorder:
            self.assertTrue(len(recorder.data_buffer) == 0)

        _ = model(torch.randn(1, 1, 1, 1))
        self.assertTrue(len(recorder.data_buffer) == 0)

    def test_module_intputs(self):

        recorder = ModuleInputsRecorder('conv1')
        model = ToyModel()
        recorder.initialize(model)

        tensor = torch.randn(1, 1, 1, 1)
        with recorder:
            self.assertTrue(recorder.recording)
            _ = model(tensor)

        conv1_input = recorder.get_record_data(data_idx=0)
        self.assertEquals(conv1_input.sum(), tensor.sum())

        with recorder:
            self.assertTrue(len(recorder.data_buffer) == 0)

        _ = model(torch.randn(1, 1, 1, 1))
        self.assertTrue(len(recorder.data_buffer) == 0)
