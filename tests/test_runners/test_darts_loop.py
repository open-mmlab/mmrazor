# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import BaseDataPreprocessor, BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader, Dataset

from mmrazor.engine import DartsEpochBasedTrainLoop  # noqa: F401
from mmrazor.engine import DartsIterBasedTrainLoop  # noqa: F401
from mmrazor.registry import DATASETS, HOOKS, MODELS


class ToyDataPreprocessor(BaseDataPreprocessor):

    def collate_data(self, data):
        data = [_data[0] for _data in data]
        inputs = [_data['inputs'].to(self._device) for _data in data]
        batch_data_samples = []
        # Model can get predictions without any data samples.
        for _data in data:
            if 'data_samples' in _data:
                batch_data_samples.append(_data['data_samples'])
        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self._device) for data_sample in batch_data_samples
        ]

        if not batch_data_samples:
            batch_data_samples = None  # type: ignore

        return inputs, batch_data_samples


@MODELS.register_module()
class ToyModel_DartsLoop(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def train_step(self, data, optim_wrapper=None):

        data1, data2 = data
        _ = self._run_forward(data1, mode='loss')
        losses = self._run_forward(data2, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        return log_vars

    def forward(self, inputs, data_samples, mode='tensor'):
        batch_inputs = torch.stack(inputs).to(self.linear1.weight.device)
        labels = torch.stack(data_samples).to(self.linear1.weight.device)
        outputs = self.linear1(batch_inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@DATASETS.register_module()
class ToyDataset_DartsLoop(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


class TestDartsLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        epoch_based_cfg = dict(
            default_scope='mmrazor',
            model=dict(type='ToyModel_DartsLoop'),
            work_dir=self.temp_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyDataset_DartsLoop'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            optim_wrapper=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            train_cfg=dict(
                type='DartsEpochBasedTrainLoop',
                max_epochs=3,
                val_interval=1,
                val_begin=2),
            custom_hooks=[],
            default_hooks=dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=True),
                sampler_seed=dict(type='DistSamplerSeedHook')),
            launcher='none',
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
        )
        self.epoch_based_cfg = Config(epoch_based_cfg)
        self.epoch_based_cfg.train_cfg['mutator_dataloader'] = \
            self.epoch_based_cfg.train_dataloader
        self.iter_based_cfg = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset_DartsLoop'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.train_cfg = dict(
            type='DartsIterBasedTrainLoop',
            mutator_dataloader=self.iter_based_cfg.train_dataloader,
            max_iters=12,
            val_interval=4,
            val_begin=4)
        self.iter_based_cfg.default_hooks = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False),
            sampler_seed=dict(type='DistSamplerSeedHook'))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # 1. DartsEpochBasedTrainLoop
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init1'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_train_loop(cfg.train_cfg)

        self.assertIsInstance(loop, DartsEpochBasedTrainLoop)
        self.assertIsInstance(loop.runner, Runner)
        self.assertEqual(loop.max_epochs, 3)
        self.assertEqual(loop.max_iters, 12)
        self.assertIsInstance(loop.mutator_dataloader, DataLoader)

        # 2. DartsIterBasedTrainLoop
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_init2'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_train_loop(cfg.train_cfg)

        self.assertIsInstance(loop, DartsIterBasedTrainLoop)
        self.assertIsInstance(loop.runner, Runner)
        self.assertEqual(loop.max_iters, 12)
        self.assertIsInstance(loop.mutator_dataloader, DataLoader)

    def test_run(self):
        # 1. test DartsEpochBasedTrainLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        iter_results = []
        iter_targets = [i for i in range(4 * 3)]
        batch_idx_results = []
        batch_idx_targets = [i for i in range(4)] * 3  # train and val
        val_epoch_results = []
        val_epoch_targets = [i for i in range(2, 4)]

        @HOOKS.register_module()
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

            def before_val_epoch(self, runner):
                val_epoch_results.append(runner.epoch)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train1'
        cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        runner = Runner.from_cfg(cfg)
        runner.train()

        assert isinstance(runner.train_loop, DartsEpochBasedTrainLoop)
        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_epoch_results, val_epoch_targets):
            self.assertEqual(result, target)

        # 2. test DartsIterBasedTrainLoop
        epoch_results = []
        iter_results = []
        batch_idx_results = []
        val_iter_results = []
        val_batch_idx_results = []
        iter_targets = [i for i in range(12)]
        batch_idx_targets = [i for i in range(12)]
        val_iter_targets = [i for i in range(4, 12)]
        val_batch_idx_targets = [i for i in range(4)] * 2

        @HOOKS.register_module()
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

            def before_val_iter(self, runner, batch_idx, data_batch=None):
                val_epoch_results.append(runner.iter)
                val_batch_idx_results.append(batch_idx)

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train2'
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        runner = Runner.from_cfg(cfg)
        runner.train()

        assert isinstance(runner.train_loop, DartsIterBasedTrainLoop)
        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        self.assertEqual(runner.val_interval, 4)
        self.assertEqual(runner.val_begin, 4)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_iter_results, val_iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_batch_idx_results,
                                   val_batch_idx_targets):
            self.assertEqual(result, target)
