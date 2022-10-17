# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmrazor.engine import GreedySamplerTrainLoop  # noqa: F401
from mmrazor.registry import DATASETS, METRICS, MODELS


@MODELS.register_module()
class ToyModel_GreedySamplerTrainLoop(BaseModel):

    @patch('mmrazor.models.mutators.OneShotModuleMutator')
    def __init__(self, mock_mutator):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
        self.mutator = mock_mutator

    def forward(self, inputs, data_samples, mode='tensor'):
        batch_inputs = torch.stack(inputs)
        labels = torch.stack(data_samples)
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

    def sample_subnet(self):
        return self.mutator.sample_choices()

    def set_subnet(self, subnet):
        self.mutator.set_choices(subnet)

    def export_fix_subnet(self):
        pass


@DATASETS.register_module()
class ToyDataset_GreedySamplerTrainLoop(Dataset):
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


@METRICS.register_module()
class ToyMetric_GreedySamplerTrainLoop(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestGreedySamplerTrainLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        val_dataloader = dict(
            dataset=dict(type='ToyDataset_GreedySamplerTrainLoop'),
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=3,
            num_workers=0)
        val_evaluator = dict(type='ToyMetric_GreedySamplerTrainLoop')

        iter_based_cfg = dict(
            default_scope='mmrazor',
            model=dict(type='ToyModel_GreedySamplerTrainLoop'),
            work_dir=self.temp_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyDataset_GreedySamplerTrainLoop'),
                sampler=dict(type='InfiniteSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=val_dataloader,
            optim_wrapper=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            val_evaluator=val_evaluator,
            train_cfg=dict(
                type='GreedySamplerTrainLoop',
                dataloader_val=val_dataloader,
                evaluator=val_evaluator,
                max_iters=12,
                val_interval=2,
                score_key='acc',
                constraints_range=None,
                num_candidates=4,
                num_samples=2,
                top_k=2,
                prob_schedule='linear',
                schedule_start_iter=4,
                schedule_end_iter=10,
                init_prob=0.,
                max_prob=0.8),
            val_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=False),
                sampler_seed=dict(type='DistSamplerSeedHook')),
            launcher='none',
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
        )
        self.iter_based_cfg = Config(iter_based_cfg)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_init_GreedySamplerTrainLoop'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_train_loop(cfg.train_cfg)
        self.assertIsInstance(loop, GreedySamplerTrainLoop)

    def test_update_cur_prob(self):
        # prob_schedule = linear
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_update_cur_prob1'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_train_loop(cfg.train_cfg)

        loop.update_cur_prob(loop.schedule_end_iter - 1)
        self.assertGreater(loop.max_prob, loop.cur_prob)
        loop.update_cur_prob(loop.schedule_end_iter + 1)
        self.assertEqual(loop.max_prob, loop.cur_prob)

        # prob_schedule = consine
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_update_cur_prob2'
        cfg.train_cfg.prob_schedule = 'consine'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_train_loop(cfg.train_cfg)

        loop.update_cur_prob(loop.schedule_end_iter - 1)
        self.assertGreater(loop.max_prob, loop.cur_prob)
        loop.update_cur_prob(loop.schedule_end_iter + 1)
        self.assertEqual(loop.max_prob, loop.cur_prob)

    def test_sample_subnet(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_sample_subnet'
        runner = Runner.from_cfg(cfg)
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        runner.model.sample_subnet = MagicMock(return_value=fake_subnet)
        loop = runner.build_train_loop(cfg.train_cfg)
        loop.cur_prob = loop.max_prob
        self.assertEqual(len(loop.top_k_candidates), 0)

        loop._iter = loop.val_interval
        subnet = loop.sample_subnet()
        self.assertEqual(subnet, fake_subnet)
        self.assertEqual(len(loop.top_k_candidates), loop.top_k)

    def test_run(self):
        # test run with _check_constraints
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_run1'
        runner = Runner.from_cfg(cfg)
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        runner.model.sample_subnet = MagicMock(return_value=fake_subnet)
        loop = runner.build_train_loop(cfg.train_cfg)
        loop._check_constraints = MagicMock(return_value=(True, dict()))
        runner.train()

        self.assertEqual(runner.iter, runner.max_iters)
        assert os.path.exists(os.path.join(self.temp_dir, 'candidates.pkl'))
