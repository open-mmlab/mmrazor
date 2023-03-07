# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.logging import MessageHub

from mmrazor.engine import StopDistillHook


class TestStopDistillHook(TestCase):

    def setUp(self):
        self.stop_epoch = 5
        self.hook = StopDistillHook(stop_epoch=self.stop_epoch)
        runner = Mock()
        runner.model = nn.Module()
        runner.model.register_buffer('distillation_stopped',
                                     torch.tensor([False]))

        runner.epoch = 0
        self.runner = runner

        message_hub = dict(name='test')
        self.message_hub = MessageHub.get_instance(**message_hub)

    def test_before_train_epoch(self):
        max_epochs = 10
        target = [False] * 5 + [True] * 5
        for epoch in range(max_epochs):
            self.hook.before_train_epoch(self.runner)
            self.assertEquals(self.runner.model.distillation_stopped,
                              target[epoch])
            self.runner.epoch += 1

            if not self.runner.model.distillation_stopped:
                self.message_hub.update_scalar('distill.loss', random.random())

            if self.runner.model.distillation_stopped:
                self.assertNotIn('distill.loss', self.message_hub.log_scalars)
            else:
                self.assertIn('distill.loss', self.message_hub.log_scalars)
