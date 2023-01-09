# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from typing import Dict, List, Tuple, Union
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.structures import ClsDataSample
from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset

from mmrazor.engine import AutoSlimGreedySearchLoop
from mmrazor.models.algorithms import AutoSlim
from mmrazor.registry import LOOPS

MUTATOR_TYPE = Union[torch.nn.Module, Dict]
DISTILLER_TYPE = Union[torch.nn.Module, Dict]


def collate_fn(data_batch):
    return data_batch[0]


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = [torch.randn(2, 3, 4, 4)] * 4
    label = [[ClsDataSample().set_gt_label(torch.randint(0, 1000, (2, )))]
             for _ in range(4)]

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


ARCHITECTURE_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

MUTATOR_CFG = dict(
    type='OneShotChannelMutator',
    channel_unit_cfg=dict(
        type='OneShotMutableChannelUnit',
        default_args=dict(
            candidate_choices=list(i / 12 for i in range(2, 13)),
            choice_mode='ratio')))

DISTILLER_CFG = dict(
    type='ConfigurableDistiller',
    teacher_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    student_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    distill_losses=dict(
        loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
    loss_forward_mappings=dict(
        loss_kl=dict(
            preds_S=dict(recorder='fc', from_student=True),
            preds_T=dict(recorder='fc', from_student=False))))


class ToyDataPreprocessor(torch.nn.Module):

    def forward(
            self,
            data: Dict,
            training: bool = True) -> Tuple[torch.Tensor, List[ClsDataSample]]:
        return data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 100, 3)

    def forward(self, x):
        out = F.conv2d(
            x,
            weight=self.conv.weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups)
        return out


class ToyRunner:

    @property
    def distributed(self):
        pass

    @property
    def rank(self):
        pass

    @property
    def epoch(self):
        pass

    @property
    def work_dir(self):
        pass

    def model(self):
        pass

    def logger(self):
        pass

    def call_hook(self, fn_name: str):
        pass

    def visualizer(self):
        pass


class TestAutoSlimGreedySearchLoop(TestCase):
    device: str = 'cpu'

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        train_cfg = dict(type='AutoSlimGreedySearchLoop', target_flops=(700, ))
        self.train_cfg = Config(train_cfg)
        self.runner = MagicMock(spec=ToyRunner)
        self.runner.model = self.prepare_model(MUTATOR_CFG, DISTILLER_CFG,
                                               ARCHITECTURE_CFG)
        self.runner.distributed = False
        self.dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        self.evaluator = MagicMock()

    def prepare_model(self,
                      mutator_cfg: MUTATOR_TYPE = MUTATOR_CFG,
                      distiller_cfg: DISTILLER_TYPE = DISTILLER_CFG,
                      architecture_cfg: Dict = ARCHITECTURE_CFG,
                      num_random_samples: int = 2) -> AutoSlim:
        model = AutoSlim(
            mutator=mutator_cfg,
            distiller=distiller_cfg,
            architecture=architecture_cfg,
            data_preprocessor=ToyDataPreprocessor(),
            num_random_samples=num_random_samples)
        model.to(self.device)

        return model

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self) -> None:
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, AutoSlimGreedySearchLoop)

    def test_run(self):
        # test_run_epoch: distributed == False
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1
        self.runner.distributed = False
        self.runner.work_dir = self.temp_dir
        loop.run()
        self.assertEqual(len(loop.searched_subnet), 1)
