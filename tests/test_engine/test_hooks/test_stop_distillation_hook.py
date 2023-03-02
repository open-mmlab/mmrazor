# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmrazor.engine import StopDistillHook


class TestStopDistillHook(TestCase):

    def setUp(self):
        self.hook = StopDistillHook(stop_epoch=5)
        runner = Mock()
        runner.model = Mock()
        runner.model.distillation_stopped = False

        runner.epoch = 0
        self.runner = runner

    def test_before_train_epoch(self):
        max_epochs = 10
        target = [False] * 5 + [True] * 5
        for epoch in range(max_epochs):
            self.hook.before_train_epoch(self.runner)
            self.assertEquals(self.runner.model.distillation_stopped,
                              target[epoch])
            self.runner.epoch += 1
