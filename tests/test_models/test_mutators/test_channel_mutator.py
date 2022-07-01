# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from os.path import dirname

import mmcv.fileio
import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch import Tensor, nn
from torch.nn import Module

from mmrazor import digit_version
from mmrazor.models.architectures.dynamic_op import (build_dynamic_bn,
                                                     build_dynamic_conv2d)
from mmrazor.models.mutables import SlimmableChannelMutable
from mmrazor.models.mutators import (OneShotChannelMutator,
                                     SlimmableChannelMutator)
from mmrazor.registry import MODELS

ONESHOT_MUTATOR_CFG = dict(
    type='OneShotChannelMutator',
    tracer_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')),
    mutable_cfg=dict(
        type='RatioChannelMutable',
        candidate_choices=[
            1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0
        ]))

ONESHOT_MUTATOR_CFG_WITHOUT_TRACER = dict(
    type='OneShotChannelMutator',
    mutable_cfg=dict(
        type='RatioChannelMutable',
        candidate_choices=[
            1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0
        ]))


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

    def __init__(self, mutable_cfg) -> None:
        super().__init__()

        self.dynamic_op1 = build_dynamic_conv2d(
            nn.Conv2d(3, 8, 1), 'dynamic_op1', mutable_cfg, mutable_cfg)
        self.dynamic_bn1 = build_dynamic_bn(
            nn.BatchNorm2d(8), 'dynamic_bn1', mutable_cfg)
        self.dynamic_op2 = build_dynamic_conv2d(
            nn.Conv2d(8, 8, 1), 'dynamic_op2', mutable_cfg, mutable_cfg)
        self.dynamic_bn2 = build_dynamic_bn(
            nn.BatchNorm2d(8), 'dynamic_bn2', mutable_cfg)
        self.dynamic_op3 = build_dynamic_conv2d(
            nn.Conv2d(8, 8, 1), 'dynamic_op3', mutable_cfg, mutable_cfg)
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
        for key, val in mutator.search_groups.items():
            print(key, val)
        # print(mutator.search_groups)
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
        ONESHOT_MUTATOR_CFG_WITHOUT_TRACER['mutable_cfg'])
    _test(dynamic_model)


def test_slimmable_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)

    root_path = dirname(dirname(dirname(__file__)))
    channel_cfgs = [
        os.path.join(root_path, 'data/subnet1.yaml'),
        os.path.join(root_path, 'data/subnet2.yaml')
    ]
    channel_cfgs = [mmcv.fileio.load(path) for path in channel_cfgs]

    mutator = SlimmableChannelMutator(
        mutable_cfg=dict(type='SlimmableChannelMutable'),
        channel_cfgs=channel_cfgs)

    model = ResBlock()
    mutator.prepare_from_supernet(model)
    mutator.switch_choices(0)
    for name, module in model.named_modules():
        if isinstance(module, SlimmableChannelMutable):
            assert module.current_choice == 0
    _ = model(imgs)

    mutator.switch_choices(1)
    for name, module in model.named_modules():
        if isinstance(module, SlimmableChannelMutable):
            assert module.current_choice == 1
    _ = model(imgs)
