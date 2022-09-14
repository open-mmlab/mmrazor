# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict
from torch import nn
from toy_mod import Toy

from mmrazor.models.task_modules import RecorderManager


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.toy = Toy()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.toy.toy_func()


class TestRecorderManager(TestCase):

    def test_init(self):

        manager = RecorderManager()
        self.assertEquals(len(manager.recorders), 0)

        recorders = ConfigDict(
            r1=dict(type='ModuleOutputs', source='conv1'),
            r2=dict(type='MethodOutputs', source='toy_mod.Toy.toy_func'),
        )
        manager = RecorderManager(recorders)
        model = ToyModel()
        manager.initialize(model)

    def test_context_manager(self):

        recorders = ConfigDict(
            r1=dict(type='ModuleOutputs', source='conv2'),
            r2=dict(type='MethodOutputs', source='toy_mod.Toy.toy_func'),
        )
        manager = RecorderManager(recorders)
        model = ToyModel()
        manager.initialize(model)

        self.assertEquals(manager.get_recorder('r1'), manager.recorders['r1'])
        self.assertEquals(manager.get_recorder('r2'), manager.recorders['r2'])

        with manager:
            res = model(torch.ones(1, 1, 1, 1))

        method_outputs = manager.recorders['r2'].get_record_data()
        conv2_outputs = manager.recorders['r1'].get_record_data()

        self.assertEquals(res.sum(), method_outputs + conv2_outputs.sum())
