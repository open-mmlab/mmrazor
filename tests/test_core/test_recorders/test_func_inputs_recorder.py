# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.evaluator import Evaluator
from mmengine.hooks import EMAHook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel, ExponentialMovingAverage
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmrazor.models.task_modules import FunctionInputsRecorder, RecorderManager


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # test FunctionInputsRecorder when ema_hook is used
        recorders_cfg = dict(
            out=dict(type='FunctionInputs', source='toy_mod.toy_func'))
        self.recorders = RecorderManager(recorders_cfg)
        self.recorders.initialize(self)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        with self.recorders:
            outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class DummyDataset(Dataset):
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


class TestFuncInputsRecorder(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_context_manager(self):
        from toy_mod import execute_toy_func2 as execute_toy_func

        recorder = FunctionInputsRecorder('toy_mod.toy_func2')
        recorder.initialize()

        with recorder:
            execute_toy_func(1, 2)
            execute_toy_func(1, b=2)
            execute_toy_func(b=2, a=1)

        self.assertTrue(
            recorder.get_record_data(record_idx=0, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=0, data_idx=1) == 2)

        self.assertTrue(
            recorder.get_record_data(record_idx=1, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=1, data_idx=1) == 2)

        self.assertTrue(
            recorder.get_record_data(record_idx=2, data_idx=0) == 1)
        self.assertTrue(
            recorder.get_record_data(record_idx=2, data_idx=1) == 2)

    def test_ema_hook(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel().to(device)
        evaluator = Evaluator([])
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmrazor',
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', )],
            experiment_name='test_func_inputs_recorder')
        runner.train()
        for hook in runner.hooks:
            if isinstance(hook, EMAHook):
                self.assertTrue(
                    isinstance(hook.ema_model, ExponentialMovingAverage))

        self.assertTrue(
            osp.exists(osp.join(self.temp_dir.name, 'epoch_2.pth')))
        checkpoint = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))
        self.assertTrue('ema_state_dict' in checkpoint)
        self.assertTrue(checkpoint['ema_state_dict']['steps'] == 8)
