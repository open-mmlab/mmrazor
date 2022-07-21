# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from os.path import dirname

import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch import Tensor, nn
from torch.nn import Module

from mmrazor import digit_version
from mmrazor.models.mutables import OneShotMutableChannel
from mmrazor.models.mutators import (OneShotChannelMutator,
                                     SlimmableChannelMutator)
from mmrazor.models.mutators.utils import (dynamic_bn_converter,
                                           dynamic_conv2d_converter)
from mmrazor.registry import MODELS
from .utils import load_and_merge_channel_cfgs

MUTABLE_CFG = dict(
    type='OneShotMutableChannel',
    candidate_choices=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0],
    candidate_mode='ratio')
MUTABLE_CFGS = dict(
    in_channels=MUTABLE_CFG,
    out_channels=MUTABLE_CFG,
    num_features=MUTABLE_CFG)

ONESHOT_MUTATOR_CFG = dict(
    type='OneShotChannelMutator',
    tracer_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')),
    global_mutable_cfgs=MUTABLE_CFGS)

ONESHOT_MUTATOR_CFG_WITHOUT_TRACER = dict(
    type='OneShotChannelMutator', global_mutable_cfgs=MUTABLE_CFGS)


class MultiConcatModel(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(16, 8, 1)
        self.op4 = nn.Conv2d(3, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.op3(cat1)
        x4 = self.op4(x)
        output = torch.cat([x3, x4], dim=1)

        return output


class MultiConcatModel2(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.op3 = nn.Conv2d(3, 8, 1)
        self.op4 = nn.Conv2d(24, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.op1(x)
        x2 = self.op2(x)
        x3 = self.op3(x)
        cat1 = torch.cat([x1, x2], dim=1)
        cat2 = torch.cat([cat1, x3], dim=1)
        output = self.op4(cat2)

        return output


class ConcatModel(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(3, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(16, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x))
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.op3(cat1)

        return x3


class ResBlock(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = nn.Conv2d(3, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.op2 = nn.Conv2d(8, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.op3 = nn.Conv2d(8, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bn1(self.op1(x))
        x2 = self.bn2(self.op2(x1))
        x3 = self.op3(x2 + x1)
        return x3


class DynamicResBlock(Module):

    def __init__(self, mutable_cfgs) -> None:
        super().__init__()

        self.dynamic_op1 = dynamic_conv2d_converter(
            nn.Conv2d(3, 8, 1), mutable_cfgs)
        self.dynamic_bn1 = dynamic_bn_converter(
            nn.BatchNorm2d(8), mutable_cfgs)
        self.dynamic_op2 = dynamic_conv2d_converter(
            nn.Conv2d(8, 8, 1), mutable_cfgs)
        self.dynamic_bn2 = dynamic_bn_converter(
            nn.BatchNorm2d(8), mutable_cfgs)
        self.dynamic_op3 = dynamic_conv2d_converter(
            nn.Conv2d(8, 8, 1), mutable_cfgs)
        self._add_link()

    def _add_link(self):
        op1_mutable_out = self.dynamic_op1.mutable_out
        bn1_mutable_out = self.dynamic_bn1.mutable_out

        op2_mutable_in = self.dynamic_op2.mutable_in
        op2_mutable_out = self.dynamic_op2.mutable_out
        bn2_mutable_out = self.dynamic_bn2.mutable_out

        op3_mutable_in = self.dynamic_op3.mutable_in

        bn1_mutable_out.register_same_mutable(op1_mutable_out)
        op1_mutable_out.register_same_mutable(bn1_mutable_out)

        op2_mutable_in.register_same_mutable(bn1_mutable_out)
        bn1_mutable_out.register_same_mutable(op2_mutable_in)

        bn2_mutable_out.register_same_mutable(op2_mutable_out)
        op2_mutable_out.register_same_mutable(bn2_mutable_out)

        op3_mutable_in.register_same_mutable(bn1_mutable_out)
        bn1_mutable_out.register_same_mutable(op3_mutable_in)

        op3_mutable_in.register_same_mutable(bn2_mutable_out)
        bn2_mutable_out.register_same_mutable(op3_mutable_in)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.dynamic_bn1(self.dynamic_op1(x))
        x2 = self.dynamic_bn2(self.dynamic_op2(x1))
        x3 = self.dynamic_op3(x2 + x1)
        return x3


@unittest.skipIf(
    digit_version(torch.__version__) == digit_version('1.8.1'),
    'PyTorch version 1.8.1 is not supported by the Backward Tracer.')
def test_oneshot_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)

    def _test(model):
        mutator.prepare_from_supernet(model)
        assert hasattr(mutator, 'name2module')

        # test set_min_choices
        mutator.set_min_choices()
        for mutables in mutator.search_groups.values():
            for mutable in mutables:
                # 1 / 8 is the minimum candidate ratio
                assert mutable.current_choice == round(1 / 8 *
                                                       mutable.num_channels)

        # test set_max_channel
        mutator.set_max_choices()
        for mutables in mutator.search_groups.values():
            for mutable in mutables:
                # 1.0 is the maximum candidate ratio
                assert mutable.current_choice == round(1. *
                                                       mutable.num_channels)

        # test making groups logic
        choice_dict = mutator.sample_choices()
        assert isinstance(choice_dict, dict)
        mutator.set_choices(choice_dict)
        model(imgs)

    mutator: OneShotChannelMutator = MODELS.build(ONESHOT_MUTATOR_CFG)
    with pytest.raises(RuntimeError):
        _ = mutator.search_groups
    with pytest.raises(RuntimeError):
        _ = mutator.name2module

    _test(ResBlock())
    _test(MultiConcatModel())
    _test(MultiConcatModel2())
    _test(nn.Sequential(nn.BatchNorm2d(3)))

    mutator: OneShotChannelMutator = MODELS.build(
        ONESHOT_MUTATOR_CFG_WITHOUT_TRACER)
    dynamic_model = DynamicResBlock(
        ONESHOT_MUTATOR_CFG_WITHOUT_TRACER['global_mutable_cfgs'])
    _test(dynamic_model)


def test_slimmable_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)

    root_path = dirname(dirname(dirname(__file__)))
    channel_cfg_paths = [
        os.path.join(root_path, 'data/subnet1.yaml'),
        os.path.join(root_path, 'data/subnet2.yaml')
    ]

    mutator = SlimmableChannelMutator(
        global_mutable_cfgs=dict(
            in_channels=dict(type='OneShotMutableChannel'),
            out_channels=dict(type='OneShotMutableChannel'),
            num_features=dict(type='OneShotMutableChannel')),
        channel_cfgs=load_and_merge_channel_cfgs(channel_cfg_paths),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss')))

    model = ResBlock()
    mutator.prepare_from_supernet(model)
    mutator.switch_choices(0)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[0]
    _ = model(imgs)

    mutator.switch_choices(1)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[1]
    _ = model(imgs)

    channel_cfg_paths = [
        os.path.join(root_path, 'data/concat_subnet1.yaml'),
        os.path.join(root_path, 'data/concat_subnet2.yaml')
    ]

    mutator = SlimmableChannelMutator(
        global_mutable_cfgs=dict(
            in_channels=dict(type='OneShotMutableChannel'),
            out_channels=dict(type='OneShotMutableChannel'),
            num_features=dict(type='OneShotMutableChannel')),
        channel_cfgs=load_and_merge_channel_cfgs(channel_cfg_paths),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss')))

    model = ConcatModel()

    mutator.prepare_from_supernet(model)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert len(module.choices) == 2

    mutator.switch_choices(0)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[0]
    _ = model(imgs)

    mutator.switch_choices(1)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[1]
    _ = model(imgs)
