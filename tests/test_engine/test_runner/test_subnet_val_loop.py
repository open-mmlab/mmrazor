# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import pytest

from mmrazor.engine.runner import SubnetValLoop


class TestSubnetValLoop(TestCase):

    def test_subnet_val_loop(self):
        runner = MagicMock()
        runner.distributed = False
        runner.model = MagicMock()
        dataloader = MagicMock()
        evaluator = [MagicMock()]
        fix_subnet_kinds = ['max', 'min', 'random']
        loop = SubnetValLoop(
            runner, dataloader, evaluator, fix_subnet_kinds=fix_subnet_kinds)

        loop.estimator = MagicMock()
        loop.estimator.estimate.return_value = dict(flops=10)

        runner.train_dataloader = MagicMock()
        with patch.object(loop, '_evaluate_once') as evaluate_mock:
            evaluate_mock.return_value = dict(acc=10)
            all_metrics = dict()
            all_metrics['max_subnet.acc'] = 10
            all_metrics['min_subnet.acc'] = 10
            all_metrics['random_subnet.acc'] = 10
            loop.run()
            loop.runner.call_hook.assert_has_calls([
                call('before_val'),
                call('before_val_epoch'),
                call('after_val_epoch', metrics=all_metrics),
                call('after_val')
            ])
            evaluate_mock.assert_has_calls([call(), call(), call()])

        runner.dataloader = MagicMock()
        runner.dataloader.dataset = MagicMock()
        loop.dataloader.__iter__.return_value = ['data_batch1']
        with patch.object(loop,
                          'calibrate_bn_statistics') as calibration_bn_mock:
            with patch.object(loop, 'run_iter') as run_iter_mock:
                eval_result = dict(acc=10)
                loop.evaluator.evaluate.return_value = eval_result
                result = loop._evaluate_once()
                calibration_bn_mock.assert_called_with(
                    runner.train_dataloader, loop.calibrate_sample_num)
                runner.model.eval.assert_called()
                run_iter_mock.assert_called_with(0, 'data_batch1')
                loop.evaluator.evaluate.assert_called_with(
                    len(runner.dataloader.dataset))
                assert result == eval_result
                loop.estimator.estimate.assert_called()

    def test_invalid_kind(self):
        runner = MagicMock()
        runner.distributed = False
        runner.model = MagicMock()
        dataloader = MagicMock()
        evaluator = [MagicMock()]
        fix_subnet_kinds = ['invalid']
        loop = SubnetValLoop(
            runner,
            dataloader,
            evaluator,
            fix_subnet_kinds=fix_subnet_kinds,
            estimator_cfg=None)
        with pytest.raises(NotImplementedError):
            loop.run()

    def test_subnet_val_loop_with_invalid_value(self):
        runner = MagicMock()
        runner.model.module = MagicMock()
        runner.model.module.__setattr__('sample_kinds', None)
        del runner.model.module.sample_kinds
        dataloader = MagicMock()
        evaluator = [MagicMock()]
        fix_subnet_kinds = []
        with pytest.raises(ValueError):
            SubnetValLoop(
                runner,
                dataloader,
                evaluator,
                fix_subnet_kinds=fix_subnet_kinds,
                estimator_cfg=None)
