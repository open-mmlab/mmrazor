# Copyright (c) OpenMMLab. All rights reserved.
import math
from unittest import TestCase

from mmrazor.models.task_modules import (CosineAnnealingLossWeightScheduler,
                                         LinearLossWeightScheduler,
                                         LossWeightSchedulerManager,
                                         MultiStepLossWeightScheduler)


class TestLossWeightScheduler(TestCase):

    def _test_scheduler_value(self, scheduler_manager, targets, epochs=10):
        schedulers = scheduler_manager.schedulers
        assert isinstance(schedulers, list)

        intervals = [(scheduler.begin, scheduler.end)
                     for scheduler in schedulers]
        intervals = sorted(intervals, key=lambda x: x[0])
        milestones = []
        for begin, end in intervals:
            if not milestones:
                milestones.append(end)
            elif end > milestones[-1]:
                milestones.append(end)

        for epoch in range(epochs):
            for scheduler in schedulers:
                if epoch in milestones:
                    scheduler_manager.base_value = scheduler_manager.cur_value

                base_value = scheduler_manager.base_value
                cur_value = scheduler_manager.cur_value
                multiplier = scheduler.get_multiplier(base_value, cur_value,
                                                      epoch)
                if multiplier is not None:
                    scheduler_manager.cur_value = multiplier
                    break
            self.assertAlmostEqual(scheduler_manager.cur_value, targets[epoch])

    def test_cos_anneal_scheduler(self):
        with self.assertRaises(AssertionError):
            CosineAnnealingLossWeightScheduler(
                begin=0, end=12, eta_min=0, eta_min_ratio=0.1)

        eta_min = 0.
        epochs = 12
        targets = [
            eta_min + (1. - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        scheduler = CosineAnnealingLossWeightScheduler(
            eta_min=0., begin=0, end=12)
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_multi_step_scheduler(self):
        # loss weight = 1.     if epoch < 2
        # loss weight = 0.1    if 2 <= epoch < 5
        # loss weight = 0.01   if 5 <= epoch < 9
        # loss weight = 0.001  if epoch >= 9
        epochs = 10
        targets = [1.] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001] * 3

        scheduler = MultiStepLossWeightScheduler(
            gamma=0.1, milestones=[2, 5, 9])
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_linear_scheduler(self):
        with self.assertRaises(ValueError):
            LinearLossWeightScheduler(start_factor=10, end=900)
        with self.assertRaises(ValueError):
            LinearLossWeightScheduler(start_factor=-1, end=900)
        with self.assertRaises(ValueError):
            LinearLossWeightScheduler(end_factor=1.001, end=900)
        with self.assertRaises(ValueError):
            LinearLossWeightScheduler(end_factor=-0.00001, end=900)
        # lr = 0.5     if epoch == 0
        # lr = 0.625   if epoch == 1
        # lr = 0.75    if epoch == 2
        # lr = 0.875   if epoch == 3
        # lr = 1.0     if epoch >= 4
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        targets = [x * 1. for x in interpolation] + [1.] * (epochs - iters)
        scheduler = LinearLossWeightScheduler(
            start_factor=start_factor, end=iters + 1)
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_cos_anneal_scheduler_convert_iterbased(self):
        epochs = 12
        eta_min = 1e-10
        epoch_length = 11
        targets = [
            eta_min + (1. - eta_min) *
            (1 + math.cos(math.pi * x / epochs / epoch_length)) / 2
            for x in range(epochs * epoch_length)
        ]
        scheduler = CosineAnnealingLossWeightScheduler.build_iter_from_epoch(
            end=epochs, eta_min=eta_min, epoch_length=epoch_length)
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets,
                                   epochs * epoch_length)

    def test_multi_step_scheduler_convert_iterbased(self):
        # lr = 1.0     if epoch < 2
        # lr = 0.1    if 2 <= epoch < 5
        # lr = 0.01   if 5 <= epoch < 9
        # lr = 0.001   if epoch >= 9
        epochs = 10
        epoch_length = 7
        targets = [1.] * 2 * epoch_length + [0.1] * 3 * epoch_length + [
            0.01
        ] * 4 * epoch_length + [0.001] * 3 * epoch_length
        scheduler = MultiStepLossWeightScheduler.build_iter_from_epoch(
            gamma=0.1, milestones=[2, 5, 9], epoch_length=epoch_length)
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets,
                                   epochs * epoch_length)

    def test_linear_scheduler_convert_iterbased(self):
        epochs = 10
        start_factor = 1.0 / 2
        end = 5
        epoch_length = 11

        iters = end * epoch_length - 1
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        targets = [x * 1. for x in interpolation] + [1.] * (
            epochs * epoch_length - iters)
        scheduler = LinearLossWeightScheduler.build_iter_from_epoch(
            start_factor=start_factor, end=end, epoch_length=epoch_length)
        scheduler_manager = LossWeightSchedulerManager([scheduler])
        self._test_scheduler_value(scheduler_manager, targets,
                                   epochs * epoch_length)

    def test_multi_scheduler_without_overlap_linear_multi_step(self):
        # use Linear in the first 5 epochs and then use MultiStep
        epochs = 12
        targets = [0.5, 0.625, 0.75, 0.875
                   ] + [1.0] * 4 + [0.1] * 3 + [0.01] * 1
        scheduler1 = LinearLossWeightScheduler(
            start_factor=1 / 2, begin=0, end=5)
        scheduler2 = MultiStepLossWeightScheduler(
            gamma=0.1, milestones=[3, 6], begin=5, end=12)
        scheduler_manager = LossWeightSchedulerManager(
            [scheduler1, scheduler2])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_multi_scheduler_without_overlap_linear_cosine(self):
        # use Linear in the first 5 epochs and then use Cosine
        epochs = 10
        targets1 = [0.5, 0.625, 0.75, 0.875, 1.0]
        scheduler1 = LinearLossWeightScheduler(
            start_factor=1 / 2, begin=0, end=5)

        eta_min = 1e-10
        targets2 = [
            eta_min + (targets1[-1] - eta_min) *
            (1 + math.cos(math.pi * x / 5)) / 2 for x in range(5)
        ]
        scheduler2 = CosineAnnealingLossWeightScheduler(
            begin=5, end=epochs, eta_min=eta_min)

        targets = targets1 + targets2
        scheduler_manager = LossWeightSchedulerManager(
            [scheduler1, scheduler2])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_multi_scheduler_with_overlap(self):
        # use Linear at first 5 epochs together with MultiStep
        epochs = 10
        targets = [0.5, 0.625, 0.75, 0.875
                   ] + [1.0] * 2 + [0.1] * 3 + [0.01] * 1
        scheduler1 = LinearLossWeightScheduler(
            start_factor=1 / 2, begin=0, end=5)
        scheduler2 = MultiStepLossWeightScheduler(
            gamma=0.1, milestones=[3, 6, 9])
        scheduler_manager = LossWeightSchedulerManager(
            [scheduler1, scheduler2])
        self._test_scheduler_value(scheduler_manager, targets, epochs)

    def test_multi_scheduler_with_gap(self):
        # use Linear in the first 5 epochs and the last 5 epochs use MultiStep
        # no scheduler in the middle 5 epochs
        epochs = 15
        targets1 = [0.5, 0.625, 0.75, 0.875, 1.0]
        scheduler1 = LinearLossWeightScheduler(
            start_factor=1 / 2, begin=0, end=5)

        scheduler2 = MultiStepLossWeightScheduler(
            gamma=0., milestones=[0], begin=10, end=15)
        targets2 = [0.] * 5
        targets = targets1 + [targets1[-1]] * 5 + targets2
        scheduler_manager = LossWeightSchedulerManager(
            [scheduler1, scheduler2])
        self._test_scheduler_value(scheduler_manager, targets, epochs)
