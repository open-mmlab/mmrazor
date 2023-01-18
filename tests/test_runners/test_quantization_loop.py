# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.registry import DATASETS, HOOKS, METRICS, MODELS, OPTIM_WRAPPERS
from mmengine.runner import Runner
from torch.nn.intrinsic.qat import ConvBnReLU2d
from torch.utils.data import Dataset

from mmrazor import digit_version
from mmrazor.engine import (LSQEpochBasedLoop, PTQLoop, QATEpochBasedLoop,
                            QATValLoop)

try:
    from torch.ao.nn.quantized import FloatFunctional, FXFloatFunctional
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.qconfig_mapping import \
        get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    QConfigMapping = get_placeholder('torch>=1.13')
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    get_default_qconfig_mapping = get_placeholder('torch>=1.13')
    FloatFunctional = get_placeholder('torch>=1.13')
    FXFloatFunctional = get_placeholder('torch>=1.13')


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 3, 4, 4)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class MMArchitectureQuant(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.architecture = ToyModel()

    def calibrate_step(self, data):
        data = self.data_preprocessor(data, False)
        return self.architecture(**data)

    def sync_qparams(self, src_mode):
        pass

    def forward(self, inputs, data_sample, mode='tensor'):
        return self.architecture(inputs, data_sample, mode)


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        qconfig = get_default_qconfig_mapping().to_dict()['']
        self.architecture = nn.Sequential(
            ConvBnReLU2d(3, 3, 1, qconfig=qconfig))

    def forward(self, inputs, data_sample, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_sample, list):
            data_sample = torch.stack(data_sample)
        outputs = self.architecture(inputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = data_sample.sum() - outputs.sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


class ToyOptimWrapper(OptimWrapper):
    ...


class ToyMetric1(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


DEFAULT_CFG = ConfigDict(
    model=dict(type='MMArchitectureQuant'),
    train_dataloader=dict(
        dataset=dict(type='ToyDataset'),
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_size=3,
        num_workers=0),
    val_dataloader=dict(
        dataset=dict(type='ToyDataset'),
        sampler=dict(type='DefaultSampler', shuffle=False),
        batch_size=3,
        num_workers=0),
    test_dataloader=dict(
        dataset=dict(type='ToyDataset'),
        sampler=dict(type='DefaultSampler', shuffle=False),
        batch_size=3,
        num_workers=0),
    optim_wrapper=dict(
        type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
    val_evaluator=dict(type='ToyMetric1'),
    test_evaluator=dict(type='ToyMetric1'),
    train_cfg=dict(),
    val_cfg=dict(),
    test_cfg=dict(),
    custom_hooks=[],
    data_preprocessor=None,
    launcher='none',
    env_cfg=dict(dist_cfg=dict(backend='nccl')),
)


class TestQATEpochBasedLoop(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.temp_dir = tempfile.mkdtemp()
        MODELS.register_module(module=MMArchitectureQuant, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        METRICS.register_module(module=ToyMetric1, force=True)
        OPTIM_WRAPPERS.register_module(module=ToyOptimWrapper, force=True)

        default_cfg = copy.deepcopy(DEFAULT_CFG)
        default_cfg = Config(default_cfg)
        default_cfg.work_dir = self.temp_dir
        default_cfg.train_cfg = ConfigDict(
            type='mmrazor.QATEpochBasedLoop',
            max_epochs=4,
            val_begin=1,
            val_interval=1,
            disable_observer_begin=-1,
            freeze_bn_begin=-1,
            dynamic_intervals=None)
        self.default_cfg = default_cfg

    def tearDown(self):
        MODELS.module_dict.pop('MMArchitectureQuant')
        DATASETS.module_dict.pop('ToyDataset')
        METRICS.module_dict.pop('ToyMetric1')
        OPTIM_WRAPPERS.module_dict.pop('ToyOptimWrapper')

        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_init_qat_train_loop'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)
        self.assertIsInstance(runner.train_loop, QATEpochBasedLoop)

    def test_run_epoch(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_train'
        runner = Runner.from_cfg(cfg)
        runner.train()

        @HOOKS.register_module(force=True)
        class TestFreezeBNHook(Hook):

            def __init__(self, freeze_bn_begin):
                self.freeze_bn_begin = freeze_bn_begin

            def after_train_epoch(self, runner):

                def check_bn_stats(mod):
                    if isinstance(mod, ConvBnReLU2d):
                        assert mod.freeze_bn
                        assert not mod.bn.training

                if runner.train_loop._epoch + 1 >= self.freeze_bn_begin:
                    runner.model.apply(check_bn_stats)

        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_freeze_bn'
        cfg.custom_hooks = [
            dict(type='TestFreezeBNHook', priority=50, freeze_bn_begin=1)
        ]
        cfg.train_cfg.freeze_bn_begin = 1
        runner = Runner.from_cfg(cfg)
        runner.train()

        @HOOKS.register_module(force=True)
        class TestDisableObserverHook(Hook):

            def __init__(self, disable_observer_begin):
                self.disable_observer_begin = disable_observer_begin

            def after_train_epoch(self, runner):

                def check_observer_stats(mod):
                    if isinstance(mod, FakeQuantizeBase):
                        assert mod.fake_quant_enabled[0] == 0

                if runner.train_loop._epoch + 1 >= self.disable_observer_begin:
                    runner.model.apply(check_observer_stats)

        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_disable_observer'
        cfg.custom_hooks = [
            dict(
                type='TestDisableObserverHook',
                priority=50,
                disable_observer_begin=1)
        ]
        cfg.train_cfg.disable_observer_begin = 1
        runner = Runner.from_cfg(cfg)
        runner.train()


class TestLSQEpochBasedLoop(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.temp_dir = tempfile.mkdtemp()
        MODELS.register_module(module=MMArchitectureQuant, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        METRICS.register_module(module=ToyMetric1, force=True)
        OPTIM_WRAPPERS.register_module(module=ToyOptimWrapper, force=True)

        default_cfg = copy.deepcopy(DEFAULT_CFG)
        default_cfg = Config(default_cfg)
        default_cfg.work_dir = self.temp_dir
        default_cfg.train_cfg = ConfigDict(
            type='mmrazor.LSQEpochBasedLoop',
            max_epochs=4,
            val_begin=1,
            val_interval=1,
            freeze_bn_begin=-1,
            dynamic_intervals=None)
        self.default_cfg = default_cfg

    def tearDown(self):
        MODELS.module_dict.pop('MMArchitectureQuant')
        DATASETS.module_dict.pop('ToyDataset')
        METRICS.module_dict.pop('ToyMetric1')
        OPTIM_WRAPPERS.module_dict.pop('ToyOptimWrapper')

        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_init_lsq_train_loop'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)
        self.assertIsInstance(runner.train_loop, LSQEpochBasedLoop)

    def test_run_epoch(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_train'
        runner = Runner.from_cfg(cfg)
        runner.train()

        @HOOKS.register_module(force=True)
        class TestFreezeBNHook(Hook):

            def __init__(self, freeze_bn_begin):
                self.freeze_bn_begin = freeze_bn_begin

            def after_train_epoch(self, runner):

                def check_bn_stats(mod):
                    if isinstance(mod, ConvBnReLU2d):
                        assert mod.freeze_bn
                        assert not mod.bn.training

                if runner.train_loop._epoch + 1 >= self.freeze_bn_begin:
                    runner.model.apply(check_bn_stats)

        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_freeze_bn'
        cfg.custom_hooks = [
            dict(type='TestFreezeBNHook', priority=50, freeze_bn_begin=1)
        ]
        cfg.train_cfg.freeze_bn_begin = 1
        runner = Runner.from_cfg(cfg)
        runner.train()


class TestQATValLoop(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.temp_dir = tempfile.mkdtemp()
        MODELS.register_module(module=MMArchitectureQuant, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        METRICS.register_module(module=ToyMetric1, force=True)
        OPTIM_WRAPPERS.register_module(module=ToyOptimWrapper, force=True)

        default_cfg = copy.deepcopy(DEFAULT_CFG)
        default_cfg = Config(default_cfg)
        default_cfg.work_dir = self.temp_dir
        default_cfg.val_cfg = ConfigDict(type='mmrazor.QATValLoop')
        self.default_cfg = default_cfg

    def tearDown(self):
        MODELS.module_dict.pop('MMArchitectureQuant')
        DATASETS.module_dict.pop('ToyDataset')
        METRICS.module_dict.pop('ToyMetric1')
        OPTIM_WRAPPERS.module_dict.pop('ToyOptimWrapper')

        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_init_qat_val_loop'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)
        self.assertIsInstance(runner.val_loop, QATValLoop)

    def test_run(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_qat_val'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optim_wrapper')
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
        runner = Runner.from_cfg(cfg)
        runner.val()


class TestPTQLoop(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.temp_dir = tempfile.mkdtemp()
        MODELS.register_module(module=MMArchitectureQuant, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        METRICS.register_module(module=ToyMetric1, force=True)
        OPTIM_WRAPPERS.register_module(module=ToyOptimWrapper, force=True)

        default_cfg = copy.deepcopy(DEFAULT_CFG)
        default_cfg = Config(default_cfg)
        default_cfg.work_dir = self.temp_dir
        # save_checkpoint in PTQLoop need train_dataloader
        default_cfg.train_cfg = ConfigDict(by_epoch=True, max_epochs=3)
        default_cfg.test_cfg = ConfigDict(
            type='mmrazor.PTQLoop',
            calibrate_dataloader=default_cfg.train_dataloader,
            calibrate_steps=32)
        self.default_cfg = default_cfg

    def tearDown(self):
        MODELS.module_dict.pop('MMArchitectureQuant')
        DATASETS.module_dict.pop('ToyDataset')
        METRICS.module_dict.pop('ToyMetric1')
        OPTIM_WRAPPERS.module_dict.pop('ToyOptimWrapper')

        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_init_ptq_loop'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)
        self.assertIsInstance(runner.test_loop, PTQLoop)

    def test_run(self):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.experiment_name = 'test_ptq_run'
        runner = Runner.from_cfg(cfg)
        runner.test()
