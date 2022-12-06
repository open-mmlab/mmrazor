# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmrazor.engine import ItePruneValLoop
from mmrazor.registry import DATASETS, METRICS, MODELS


def collate_fn(data_batch):
    return data_batch


MODEL_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


@MODELS.register_module()
class ToyModel_ItePruneValLoop(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        inputs = torch.stack(inputs)
        labels = torch.stack(data_samples)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = torch.mean(outputs, (1, 2, 3))

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
class ToyDataset_ItePruneValLoop(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 3, 32, 32)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


@METRICS.register_module()
class ToyMetric_ItePruneValLoop(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestItePruneValLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        val_dataloader = dict(
            dataset=dict(type='ToyDataset_ItePruneValLoop'),
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=3,
            num_workers=0)
        val_evaluator = dict(type='ToyMetric_ItePruneValLoop')

        val_loop_cfg = dict(
            default_scope='mmrazor',
            model=dict(type='ToyModel_ItePruneValLoop'),
            work_dir=self.temp_dir,
            val_dataloader=val_dataloader,
            val_evaluator=val_evaluator,
            val_cfg=dict(type='ItePruneValLoop'),
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
        self.val_loop_cfg = Config(val_loop_cfg)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.val_loop_cfg)
        cfg.experiment_name = 'test_init_self'
        runner = Runner.from_cfg(cfg)
        loop = runner.build_val_loop(cfg.val_cfg)

        self.assertIsInstance(loop, ItePruneValLoop)
