# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmcls.models import *  # noqa: F401,F403
from torch import Tensor
from torch.nn import Module

from mmrazor.models import OneShotModuleMutator, OneShotMutableModule
from mmrazor.registry import MODELS

MODEL_CFG = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='mmcls.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

MUTATOR_CFG = dict(type='OneShotModuleMutator')

MUTABLE_CFG = dict(
    type='OneShotMutableOP',
    candidates=dict(
        choice1=dict(
            type='MBBlock',
            in_channels=3,
            out_channels=3,
            expand_ratio=1,
            kernel_size=3),
        choice2=dict(
            type='MBBlock',
            in_channels=3,
            out_channels=3,
            expand_ratio=1,
            kernel_size=5),
        choice3=dict(
            type='MBBlock',
            in_channels=3,
            out_channels=3,
            expand_ratio=1,
            kernel_size=7)))


def test_one_shot_mutator_normal_model() -> None:
    model = MODELS.build(MODEL_CFG)
    mutator: OneShotModuleMutator = MODELS.build(MUTATOR_CFG)

    assert mutator.mutable_class_type == OneShotMutableModule

    with pytest.raises(RuntimeError):
        _ = mutator.search_groups

    mutator.prepare_from_supernet(model)
    assert len(mutator.search_groups) == 0
    assert len(mutator.sample_choices()) == 0


class _SearchableModel(Module):

    def __init__(self) -> None:
        super().__init__()

        self.op1 = MODELS.build(MUTABLE_CFG)
        self.op2 = MODELS.build(MUTABLE_CFG)
        self.op3 = MODELS.build(MUTABLE_CFG)

    def forward(self, x: Tensor) -> Tensor:
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)

        return x


def test_one_shot_mutator_mutable_model() -> None:
    model = _SearchableModel()
    mutator: OneShotModuleMutator = MODELS.build(MUTATOR_CFG)

    mutator.prepare_from_supernet(model)
    assert list(mutator.search_groups.keys()) == [0, 1, 2]

    random_choices = mutator.sample_choices()
    assert list(random_choices.keys()) == [0, 1, 2]
    for choice in random_choices.values():
        assert choice in ['choice1', 'choice2', 'choice3']

    custom_group = [['op1', 'op2'], ['op3']]
    mutator_cfg = copy.deepcopy(MUTATOR_CFG)
    mutator_cfg.update({'custom_group': custom_group})
    mutator = MODELS.build(mutator_cfg)

    mutator.prepare_from_supernet(model)
    assert list(mutator.search_groups.keys()) == [0, 1]

    random_choices = mutator.sample_choices()
    assert list(random_choices.keys()) == [0, 1]
    for choice in random_choices.values():
        assert choice in ['choice1', 'choice2', 'choice3']

    mutator.set_choices(random_choices)

    custom_group.append(['op4'])
    mutator_cfg = copy.deepcopy(MUTATOR_CFG)
    mutator_cfg.update({'custom_group': custom_group})
    mutator = MODELS.build(mutator_cfg)
    with pytest.raises(AssertionError):
        mutator.prepare_from_supernet(model)


if __name__ == '__main__':
    pytest.main()
