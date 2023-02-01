# Copyright (c) OpenMMLab. All rights reserved.
import math
from unittest import TestCase
from unittest.mock import Mock

import torch

from mmrazor.engine import LossWeightSchedulerHook
from mmrazor.models.task_modules import (CosineAnnealingLossWeightScheduler,
                                         LinearLossWeightScheduler,
                                         LossWeightScheduler,
                                         LossWeightSchedulerManager)


class TestParamSchedulerHook(TestCase):

    def setUp(self):
        self.hook = LossWeightSchedulerHook()
        runner = Mock()
        runner.model = Mock()

        epochs = 10
        epoch_length = 7
        scheduler1 = LinearLossWeightScheduler(
            start_factor=1 / 2, begin=0, end=5)

        eta_min = 1e-10
        scheduler2 = CosineAnnealingLossWeightScheduler.build_iter_from_epoch(
            begin=5, end=epochs, eta_min=eta_min, epoch_length=epoch_length)
        runner.model.loss_weight_scheduler_manager = \
            LossWeightSchedulerManager([scheduler1, scheduler2])
        runner.epoch = 0
        runner.iter = 0
        runner.train_dataloader = [torch.rand(1)] * epoch_length
        self.runner = runner

    def reset(self):
        self.runner.model.loss_weight_scheduler_manager.cur_value = 1.
        self.runner.model.loss_weight_scheduler_manager.base_value = 1.
        self.runner.epoch = 0
        self.runner.iter = 0

    def test_before_run(self):
        self.reset()
        self.hook.before_run(self.runner)
        self.assertEquals(self.hook.milestones, [35, 70])
        schedulers = self.runner.model.loss_weight_scheduler_manager.schedulers
        for scheduler in schedulers:
            self.assertIsInstance(scheduler, LossWeightScheduler)

    def test_before_train_epoch(self):
        self.reset()
        epochs = 10
        epoch_length = 7
        targets1 = [0.5, 0.625, 0.75, 0.875, 1.0]
        targets = targets1 + [targets1[-1]] * 5
        for epoch in range(epochs):
            self.hook.before_train_epoch(self.runner)
            self.assertAlmostEqual(
                self.runner.model.loss_weight_scheduler_manager.cur_value,
                targets[epoch])
            self.runner.epoch += 1
            self.runner.iter += epoch_length

    def test_after_train_iter(self):
        self.reset()
        epochs = 10
        epoch_length = 7
        eta_min = 1e-10
        targets2 = [
            eta_min + (1.0 - eta_min) *
            (1 + math.cos(math.pi * x / 5 / epoch_length)) / 2
            for x in range(5 * epoch_length)
        ]
        targets = [1.0] * 5 * epoch_length + targets2
        for iter in range(epochs * epoch_length):
            self.hook.before_train_iter(self.runner, iter)
            self.assertAlmostEqual(
                self.runner.model.loss_weight_scheduler_manager.cur_value,
                targets[iter])
            self.runner.iter += 1
            if iter > 0 and iter % epoch_length == 0:
                self.runner.epoch += 1

    def test_train(self):
        self.reset()
        epochs = 10
        epoch_length = 7
        targets1 = [0.5, 0.625, 0.75, 0.875, 1.0]
        eta_min = 1e-10
        targets2 = [
            eta_min + (targets1[-1] - eta_min) *
            (1 + math.cos(math.pi * x / 5 / epoch_length)) / 2
            for x in range(5 * epoch_length)
        ]
        targets = []
        for num in targets1:
            targets += [num] * epoch_length
        targets += targets2
        for epoch in range(epochs):
            self.hook.before_train_epoch(self.runner)
            for iter in range(epoch_length):
                self.hook.before_train_iter(self.runner, iter)
                self.assertAlmostEqual(
                    self.runner.model.loss_weight_scheduler_manager.cur_value,
                    targets[epoch * epoch_length + iter])
                self.runner.iter += 1
            self.runner.epoch += 1
