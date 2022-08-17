# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmrazor.engine import ResourceEvaluatorLoop
from mmrazor.registry import DATASETS, METRICS, MODELS


@MODELS.register_module()
class ToyModel_EvaluatorValLoop(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        outputs = self.linear1(batch_inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_samples - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@DATASETS.register_module()
class ToyDataset_EvaluatorValLoop(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


@METRICS.register_module()
class ToyMetric_EvaluatorValLoop(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestResourceEvaluatorLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        val_dataloader = dict(
            dataset=dict(type='ToyDataset_EvaluatorValLoop'),
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=3,
            num_workers=0)
        val_evaluator = dict(type='ToyMetric_EvaluatorValLoop')
        # val_evaluator = dict(type='mmcls.Accuracy', topk=(1, ))

        val_loop_cfg = dict(
            default_scope='mmrazor',
            model=dict(type='ToyModel_EvaluatorValLoop'),
            work_dir=self.temp_dir,
            val_dataloader=val_dataloader,
            val_evaluator=val_evaluator,
            val_cfg=dict(
                type='ResourceEvaluatorLoop',
                dataloader=val_dataloader,
                evaluator=val_evaluator,
                estimator_cfg=dict(type='ResourceEstimator'),
                resource_args=dict(
                    input_shape=(1, 3, 2, 2), measure_inference=False)),
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
            env_cfg=dict(dist_cfg=dict(backend='nccl')))
        self.val_loop_cfg = Config(val_loop_cfg)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # test_init: dataloader and evaluator are instances
        cfg = copy.deepcopy(self.val_loop_cfg)
        cfg.experiment_name = 'test_init'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_val_loop(cfg.val_cfg)

        self.assertIsInstance(loop, ResourceEvaluatorLoop)

    def test_run(self):
        cfg = copy.deepcopy(self.val_loop_cfg)
        cfg.experiment_name = 'test_run'
        runner = Runner.from_cfg(cfg)
        runner.val()

        self.assertIn('val/flops', runner.message_hub.log_scalars.keys())
        self.assertIn('val/params', runner.message_hub.log_scalars.keys())
        self.assertIn('val/latency', runner.message_hub.log_scalars.keys())
