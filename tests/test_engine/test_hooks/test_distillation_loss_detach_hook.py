# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmrazor.engine import DistillationLossDetachHook


class TestDistillationLossDetachHook(TestCase):

    def setUp(self):
        self.hook = DistillationLossDetachHook(detach_epoch=5)
        runner = Mock()
        runner.model = Mock()
        runner.model.distill_loss_detach = False

        runner.epoch = 0
        # runner.max_epochs = 10
        self.runner = runner

    def test_before_train_epoch(self):
        max_epochs = 10
        target = [False] * 5 + [True] * 5
        for epoch in range(max_epochs):
            self.hook.before_train_epoch(self.runner)
            self.assertEquals(self.runner.model.distill_loss_detach,
                              target[epoch])
            self.runner.epoch += 1
