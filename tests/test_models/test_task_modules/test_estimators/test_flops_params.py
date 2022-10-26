# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import pytest
import torch
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from torch import Tensor
from torch.nn import Conv2d, Module, Parameter

from mmrazor.models import OneShotMutableModule, ResourceEstimator
from mmrazor.models.task_modules.estimators.counters import BaseCounter
from mmrazor.registry import MODELS, TASK_UTILS
from mmrazor.structures import export_fix_subnet, load_fix_subnet

_FIRST_STAGE_MUTABLE = dict(
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e1=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6'))))

_OTHER_STAGE_MUTABLE = dict(
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e3=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k5e3=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        identity=dict(type='Identity')))

ARCHSETTING_CFG = [
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride, mutable cfg.
    [16, 1, 1, _FIRST_STAGE_MUTABLE],
    [24, 2, 2, _OTHER_STAGE_MUTABLE],
    [32, 3, 2, _OTHER_STAGE_MUTABLE],
    [64, 4, 2, _OTHER_STAGE_MUTABLE],
    [96, 3, 1, _OTHER_STAGE_MUTABLE],
    [160, 3, 2, _OTHER_STAGE_MUTABLE],
    [320, 1, 1, _OTHER_STAGE_MUTABLE]
]

NORM_CFG = dict(type='BN')
BACKBONE_CFG = dict(
    type='mmrazor.SearchableMobileNet',
    first_channels=32,
    last_channels=1280,
    widen_factor=1.0,
    norm_cfg=NORM_CFG,
    arch_setting=ARCHSETTING_CFG)

estimator = ResourceEstimator()


class FoolAddConstant(Module):

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()

        self.register_parameter(
            name='p', param=Parameter(torch.tensor(p, dtype=torch.float32)))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.p


@TASK_UTILS.register_module()
class FoolAddConstantCounter(BaseCounter):

    @staticmethod
    def add_count_hook(module, input, output):
        module.__flops__ += 1000000
        module.__params__ += 700000


class FoolConv2d(Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv2d = Conv2d(3, 32, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2d(x)


class FoolConvModule(Module):

    def __init__(self) -> None:
        super().__init__()

        self.add_constant = FoolAddConstant(0.1)
        self.conv2d = FoolConv2d()

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_constant(x)

        return self.conv2d(x)


class TestResourceEstimator(TestCase):

    def sample_choice(self, model: Module) -> None:
        for module in model.modules():
            if isinstance(module, OneShotMutableModule):
                module.current_choice = module.sample_choice()

    def test_estimate(self) -> None:
        fool_conv2d = FoolConv2d()
        flops_params_cfg = dict(input_shape=(1, 3, 224, 224))
        results = estimator.estimate(
            model=fool_conv2d, flops_params_cfg=flops_params_cfg)
        flops_count = results['flops']
        params_count = results['params']

        self.assertEqual(flops_count, 44.158)
        self.assertEqual(params_count, 0.001)

        fool_conv2d = Conv2dAdaptivePadding(3, 32, 3)
        results = estimator.estimate(
            model=fool_conv2d, flops_params_cfg=flops_params_cfg)
        flops_count = results['flops']
        params_count = results['params']

        self.assertEqual(flops_count, 44.958)
        self.assertEqual(params_count, 0.001)

    def test_register_module(self) -> None:
        fool_add_constant = FoolConvModule()
        flops_params_cfg = dict(input_shape=(1, 3, 224, 224))
        results = estimator.estimate(
            model=fool_add_constant, flops_params_cfg=flops_params_cfg)
        flops_count = results['flops']
        params_count = results['params']

        self.assertEqual(flops_count, 45.158)
        self.assertEqual(params_count, 0.701)

    def test_disable_sepc_counter(self) -> None:
        fool_add_constant = FoolConvModule()
        flops_params_cfg = dict(
            input_shape=(1, 3, 224, 224),
            disabled_counters=['FoolAddConstantCounter'])
        rest_results = estimator.estimate(
            model=fool_add_constant, flops_params_cfg=flops_params_cfg)
        rest_flops_count = rest_results['flops']
        rest_params_count = rest_results['params']

        self.assertLess(rest_flops_count, 45.158)
        self.assertLess(rest_params_count, 0.701)

        fool_conv2d = Conv2dAdaptivePadding(3, 32, 3)
        flops_params_cfg = dict(
            input_shape=(1, 3, 224, 224), disabled_counters=['Conv2dCounter'])
        rest_results = estimator.estimate(
            model=fool_conv2d, flops_params_cfg=flops_params_cfg)
        rest_flops_count = rest_results['flops']
        rest_params_count = rest_results['params']

        self.assertEqual(rest_flops_count, 0)
        self.assertEqual(rest_params_count, 0)

    def test_estimate_spec_module(self) -> None:
        fool_add_constant = FoolConvModule()
        flops_params_cfg = dict(
            input_shape=(1, 3, 224, 224),
            spec_modules=['add_constant', 'conv2d'])
        results = estimator.estimate(
            model=fool_add_constant, flops_params_cfg=flops_params_cfg)
        flops_count = results['flops']
        params_count = results['params']

        self.assertEqual(flops_count, 45.158)
        self.assertEqual(params_count, 0.701)

    def test_estimate_separation_modules(self) -> None:
        fool_add_constant = FoolConvModule()
        flops_params_cfg = dict(
            input_shape=(1, 3, 224, 224), spec_modules=['add_constant'])
        results = estimator.estimate_separation_modules(
            model=fool_add_constant, flops_params_cfg=flops_params_cfg)
        self.assertGreater(results['add_constant']['flops'], 0)

        with pytest.raises(AssertionError):
            flops_params_cfg = dict(
                input_shape=(1, 3, 224, 224), spec_modules=['backbone'])
            results = estimator.estimate_separation_modules(
                model=fool_add_constant, flops_params_cfg=flops_params_cfg)

        with pytest.raises(AssertionError):
            flops_params_cfg = dict(
                input_shape=(1, 3, 224, 224), spec_modules=[])
            results = estimator.estimate_separation_modules(
                model=fool_add_constant, flops_params_cfg=flops_params_cfg)

    def test_estimate_subnet(self) -> None:
        flops_params_cfg = dict(input_shape=(1, 3, 224, 224))
        model = MODELS.build(BACKBONE_CFG)
        self.sample_choice(model)
        copied_model = copy.deepcopy(model)

        results = estimator.estimate(
            model=copied_model, flops_params_cfg=flops_params_cfg)
        flops_count = results['flops']
        params_count = results['params']

        fix_subnet = export_fix_subnet(model)
        load_fix_subnet(copied_model, fix_subnet)
        subnet_results = estimator.estimate(
            model=copied_model, flops_params_cfg=flops_params_cfg)
        subnet_flops_count = subnet_results['flops']
        subnet_params_count = subnet_results['params']

        self.assertEqual(flops_count, subnet_flops_count)
        self.assertEqual(params_count, subnet_params_count)

        # test whether subnet estimate will affect original model
        copied_model = copy.deepcopy(model)
        results_after_estimate = estimator.estimate(
            model=copied_model, flops_params_cfg=flops_params_cfg)
        flops_count_after_estimate = results_after_estimate['flops']
        params_count_after_estimate = results_after_estimate['params']

        self.assertEqual(flops_count, flops_count_after_estimate)
        self.assertEqual(params_count, params_count_after_estimate)
