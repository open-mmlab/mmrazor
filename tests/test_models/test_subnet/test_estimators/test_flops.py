# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import pytest
import torch
from torch import Tensor
from torch.nn import Conv2d, Module, Parameter

from mmrazor.models import OneShotMutable
from mmrazor.models.subnet import FlopsEstimator, export_fix_subnet
from mmrazor.registry import MODELS

_FIRST_STAGE_MUTABLE = dict(
    type='OneShotOP',
    candidate_ops=dict(
        mb_k3e1=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6'))))

_OTHER_STAGE_MUTABLE = dict(
    type='OneShotOP',
    candidate_ops=dict(
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


class FoolAddConstant(Module):

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()

        self.register_parameter(
            name='p', param=Parameter(torch.tensor(p, dtype=torch.float32)))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.p


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


class TestFlopsEstimator(TestCase):

    def sample_choice(self, model: Module) -> None:
        for module in model.modules():
            if isinstance(module, OneShotMutable):
                module.current_choice = module.sample_choice()

    def test_get_model_complexity_info(self) -> None:
        fool_conv2d = FoolConv2d()
        flops_count, params_count = FlopsEstimator.get_model_complexity_info(
            fool_conv2d, as_strings=False)

        self.assertGreater(flops_count, 0)
        self.assertGreater(params_count, 0)

    def test_register_module(self) -> None:
        fool_add_constant = FoolConvModule()
        copied_module = copy.deepcopy(fool_add_constant)
        flops_count, params_count = FlopsEstimator.get_model_complexity_info(
            copied_module, as_strings=False)

        def fool_add_constant_flops_counter_hook(add_constant_module: Module,
                                                 input: Tensor,
                                                 output: Tensor) -> None:
            add_constant_module.__flops__ = 1e6

        # test register directly
        FlopsEstimator.register_module(
            flops_counter_hook=fool_add_constant_flops_counter_hook,
            module=FoolAddConstant)
        copied_module = copy.deepcopy(fool_add_constant)
        flops_count_after_registered, params_count_after_registered = \
            FlopsEstimator.get_model_complexity_info(
                model=copied_module, as_strings=False)
        self.assertEqual(flops_count_after_registered - flops_count, 1e6)
        self.assertEqual(params_count_after_registered - params_count, 0)
        FlopsEstimator.remove_custom_module(FoolAddConstant)

        # test register using decorator
        FlopsEstimator.register_module(
            flops_counter_hook=fool_add_constant_flops_counter_hook)(
                FoolAddConstant)
        copied_module = copy.deepcopy(fool_add_constant)
        flops_count_after_registered, params_count_after_registered = \
            FlopsEstimator.get_model_complexity_info(
                model=copied_module, as_strings=False)
        self.assertEqual(flops_count_after_registered - flops_count, 1e6)
        self.assertEqual(params_count_after_registered - params_count, 0)

        FlopsEstimator.remove_custom_module(FoolAddConstant)

    def test_register_module_wrong_parameter(self) -> None:

        def fool_flops_counter_hook(module: Module, input: Tensor,
                                    output: Tensor) -> None:
            return

        with pytest.raises(TypeError):
            FlopsEstimator.register_module(
                flops_counter_hook=fool_flops_counter_hook, force=1)
        with pytest.raises(TypeError):
            FlopsEstimator.register_module(
                flops_counter_hook=fool_flops_counter_hook, module=list)
        with pytest.raises(TypeError):
            FlopsEstimator.register_module(
                flops_counter_hook=123, module=FoolAddConstant)

        # test double register
        FlopsEstimator.register_module(
            flops_counter_hook=fool_flops_counter_hook, module=FoolAddConstant)
        with pytest.raises(KeyError):
            FlopsEstimator.register_module(
                flops_counter_hook=fool_flops_counter_hook,
                module=FoolAddConstant)
        FlopsEstimator.register_module(
            flops_counter_hook=fool_flops_counter_hook,
            module=FoolAddConstant,
            force=True)

        FlopsEstimator.remove_custom_module(FoolAddConstant)

    def test_remove_custom_module(self) -> None:
        with pytest.raises(KeyError):
            FlopsEstimator.remove_custom_module(FoolAddConstant)

        def fool_flops_counter_hook(module: Module, input: Tensor,
                                    output: Tensor) -> None:
            return

        FlopsEstimator.register_module(
            flops_counter_hook=fool_flops_counter_hook, module=FoolAddConstant)

        FlopsEstimator.remove_custom_module(FoolAddConstant)

    def test_clear_custom_module(self) -> None:

        def fool_flops_counter_hook(module: Module, input: Tensor,
                                    output: Tensor) -> None:
            return

        FlopsEstimator.register_module(
            flops_counter_hook=fool_flops_counter_hook, module=FoolAddConstant)
        FlopsEstimator.register_module(
            flops_counter_hook=fool_flops_counter_hook, module=FoolConvModule)

        FlopsEstimator.clear_custom_module()
        self.assertEqual(FlopsEstimator.get_custom_modules_mapping(), {})

    def test_get_model_complexity_info_subnet(self) -> None:
        model = MODELS.build(BACKBONE_CFG)
        self.sample_choice(model)
        copied_model = copy.deepcopy(model)

        flops_count, params_count = FlopsEstimator.get_model_complexity_info(
            copied_model, as_strings=False)

        fix_subnet = export_fix_subnet(model)
        subnet_flops_count, subnet_params_count = \
            FlopsEstimator.get_model_complexity_info(
                model, fix_subnet, as_strings=False)

        self.assertEqual(flops_count, subnet_flops_count)
        self.assertGreater(params_count, subnet_params_count)

        # test whether subnet estimate will affect original model
        copied_model = copy.deepcopy(model)
        flops_count_after_estimate, params_count_after_estimate = \
            FlopsEstimator.get_model_complexity_info(
                copied_model, as_strings=False)

        self.assertEqual(flops_count, flops_count_after_estimate)
        self.assertEqual(params_count, params_count_after_estimate)
