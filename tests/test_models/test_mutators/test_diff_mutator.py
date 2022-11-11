# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch.nn as nn

from mmrazor.models import *  # noqa: F401,F403
from mmrazor.models.mutables import DiffMutableModule
from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.registry import MODELS

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='torchMaxPool2d', module=nn.MaxPool2d, force=True)
MODELS.register_module(name='torchAvgPool2d', module=nn.AvgPool2d, force=True)


class SearchableLayer(nn.Module):

    def __init__(self, mutable_cfg: dict) -> None:
        super().__init__()
        self.op1 = MODELS.build(mutable_cfg)
        self.op2 = MODELS.build(mutable_cfg)
        self.op3 = MODELS.build(mutable_cfg)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        return self.op3(x)


class SearchableModel(nn.Module):

    def __init__(self, mutable_cfg: dict) -> None:
        super().__init__()
        self.slayer1 = SearchableLayer(mutable_cfg)
        self.slayer2 = SearchableLayer(mutable_cfg)
        self.slayer3 = SearchableLayer(mutable_cfg)

    def forward(self, x):
        x = self.slayer1(x)
        x = self.slayer2(x)
        return self.slayer3(x)


class SearchableLayerAlias(nn.Module):

    def __init__(self, mutable_cfg: dict) -> None:
        super().__init__()
        mutable_cfg.update(alias='op1')
        self.op1 = MODELS.build(mutable_cfg)
        mutable_cfg.update(alias='op2')
        self.op2 = MODELS.build(mutable_cfg)
        mutable_cfg.update(alias='op3')
        self.op3 = MODELS.build(mutable_cfg)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        return self.op3(x)


class SearchableModelAlias(nn.Module):

    def __init__(self, mutable_cfg: dict) -> None:
        super().__init__()
        self.slayer1 = SearchableLayerAlias(mutable_cfg)
        self.slayer2 = SearchableLayerAlias(mutable_cfg)
        self.slayer3 = SearchableLayerAlias(mutable_cfg)

    def forward(self, x):
        x = self.slayer1(x)
        x = self.slayer2(x)
        return self.slayer3(x)


class TestDiffModuleMutator(TestCase):

    def setUp(self):
        self.MUTABLE_CFG = dict(
            type='DiffMutableOP',
            candidates=dict(
                torch_conv2d_3x3=dict(
                    type='torchConv2d',
                    kernel_size=3,
                    padding=1,
                ),
                torch_conv2d_5x5=dict(
                    type='torchConv2d',
                    kernel_size=5,
                    padding=2,
                ),
                torch_conv2d_7x7=dict(
                    type='torchConv2d',
                    kernel_size=7,
                    padding=3,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        self.MUTATOR_CFG = dict(
            type='DiffModuleMutator',
            custom_groups=[['op1'], ['op2'], ['op3']])

    def test_diff_mutator_diffop_layer(self) -> None:
        model = SearchableLayer(self.MUTABLE_CFG)
        mutator: DiffModuleMutator = MODELS.build(self.MUTATOR_CFG)

        mutator.prepare_from_supernet(model)
        assert list(mutator.search_groups.keys()) == [0, 1, 2]

    def test_diff_mutator_diffop_model(self) -> None:
        model = SearchableModel(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer3.op3'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        mutator.prepare_from_supernet(model)
        assert list(mutator.search_groups.keys()) == [0, 1, 2]

        mutator.modify_supernet_forward(mutator.arch_params)
        assert mutator.mutable_class_type == DiffMutableModule

    def test_diff_mutator_diffop_model_error(self) -> None:
        model = SearchableModel(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer3.op3_error_key'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        with pytest.raises(AssertionError):
            mutator.prepare_from_supernet(model)

    def test_diff_mutator_diffop_alias(self) -> None:
        model = SearchableModelAlias(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [['op1'], ['op2'], ['op3']]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        mutator.prepare_from_supernet(model)

        assert list(mutator.search_groups.keys()) == [0, 1, 2]

        mutator.modify_supernet_forward(mutator.arch_params)
        assert mutator.mutable_class_type == DiffMutableModule

    def test_diff_mutator_alias_module_name(self) -> None:
        """Using both alias and module name for grouping."""
        model = SearchableModelAlias(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [['op1'],
                                        [
                                            'slayer1.op2', 'slayer2.op2',
                                            'slayer3.op2'
                                        ], ['slayer1.op3', 'slayer2.op3']]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        mutator.prepare_from_supernet(model)

        assert list(mutator.search_groups.keys()) == [0, 1, 2, 3]

        mutator.modify_supernet_forward(mutator.arch_params)
        assert mutator.mutable_class_type == DiffMutableModule

    def test_diff_mutator_duplicate_keys(self) -> None:
        model = SearchableModel(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer2.op3'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        with pytest.raises(AssertionError):
            mutator.prepare_from_supernet(model)

    def test_diff_mutator_duplicate_key_alias(self) -> None:
        model = SearchableModelAlias(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['op1', 'slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer3.op3'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        with pytest.raises(AssertionError):
            mutator.prepare_from_supernet(model)

    def test_diff_mutator_illegal_key(self) -> None:
        model = SearchableModel(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['illegal_key', 'slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer3.op3'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)

        with pytest.raises(AssertionError):
            mutator.prepare_from_supernet(model)

    def test_sample_and_set_choices(self):
        model = SearchableModel(self.MUTABLE_CFG)

        mutator_cfg = self.MUTATOR_CFG.copy()
        mutator_cfg['custom_groups'] = [
            ['slayer1.op1', 'slayer2.op1', 'slayer3.op1'],
            ['slayer1.op2', 'slayer2.op2', 'slayer3.op2'],
            ['slayer1.op3', 'slayer2.op3', 'slayer3.op3'],
        ]
        mutator: DiffModuleMutator = MODELS.build(mutator_cfg)
        mutator.prepare_from_supernet(model)
        choices = mutator.sample_choices()
        mutator.set_choices(choices)
        self.assertTrue(len(choices) == 3)


if __name__ == '__main__':
    import unittest
    unittest.main()
