# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import pytest
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrazor.models.architectures.utils import mutate_conv_module
from mmrazor.models.mutables import (MutableChannelContainer, MutableValue,
                                     OneShotMutableChannel,
                                     OneShotMutableChannelUnit,
                                     OneShotMutableValue)
from mmrazor.models.mutables.mutable_module import MutableModule
from mmrazor.models.mutators import NasMutator
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
    """A searchable model with a mixed search space as follows:

    1. value search.
    2. module search.
    3. channel search.
    """

    def __init__(self, mutable_cfg: dict) -> None:
        super().__init__()

        self.first_conv = ConvModule(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=dict(type='mmrazor.BigNasConv2d'),
            norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d'))

        self.second_conv = ConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=1,
            conv_cfg=dict(type='mmrazor.BigNasConv2d'))

        self.slayer1 = SearchableLayer(mutable_cfg)
        self.slayer2 = SearchableLayer(mutable_cfg)
        self.slayer3 = SearchableLayer(mutable_cfg)

        self.register_mutables()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.slayer1(x)
        x = self.slayer2(x)
        return self.slayer3(x)

    def register_mutables(self):
        """Mutate the defined model."""
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        mutable_kernel_size = OneShotMutableValue(
            value_list=[1, 3], default_value=3)
        mutable_out_channels = OneShotMutableChannel(
            32, candidate_choices=[16, 32])
        mutate_conv_module(
            self.first_conv,
            mutable_kernel_size=mutable_kernel_size,
            mutable_out_channels=mutable_out_channels)

        # dont forget the last connection.
        MutableChannelContainer.register_mutable_channel_to_module(
            self.second_conv.conv, mutable_out_channels, False)


class TestNasMutator(unittest.TestCase):

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

        self.MUTATOR_CFG = dict(type='NasMutator')

    def test_models_with_predefined_dynamic_op(self):
        for Model in [SearchableModel]:
            with self.subTest(model=Model):
                model = SearchableModel(self.MUTABLE_CFG)
                mutator = MODELS.build(self.MUTATOR_CFG)
                assert isinstance(mutator, NasMutator)

                with pytest.raises(RuntimeError):
                    _ = mutator.search_groups
                mutator.prepare_from_supernet(model)
                assert hasattr(mutator, 'search_groups')

                with pytest.raises(AttributeError):
                    _ = mutator.arch_params
                mutator.prepare_arch_params()
                assert hasattr(mutator, 'arch_params')

                for name in mutator.search_groups.keys():
                    assert 'value' or 'channel' or 'module' in name

                self.assertEqual(len(mutator.arch_params.keys()), 9)
                for v in mutator.arch_params.values():
                    self.assertEqual(v.size()[0], 3)

                mutable_values = []
                mutable_modules = []
                for name, module in model.named_modules():
                    if isinstance(module, MutableValue):
                        mutable_values.append(name)
                    elif isinstance(module, MutableModule):
                        mutable_modules.append(name)
                    elif hasattr(module, 'source_mutables'):
                        for each_mutables in module.source_mutables:
                            if isinstance(each_mutables, MutableValue):
                                mutable_values.append(each_mutables)
                            elif isinstance(each_mutables, MutableModule):
                                mutable_modules.append(each_mutables)

                num_mutables = len(mutable_values) + \
                    len(mutable_modules) + len(mutator.mutable_units)
                self.assertEqual(len(mutator.search_groups), num_mutables)

                choices = mutator.sample_choices()
                min_choices = mutator.sample_choices(kind='min')
                max_choices = mutator.sample_choices(kind='max')

                self.assertEqual(choices.keys(), min_choices.keys())
                self.assertEqual(choices.keys(), max_choices.keys())

                with self.assertRaises(NotImplementedError):
                    _ = mutator.sample_choices(kind='mun')

                assert hasattr(mutator, 'current_choices')
                with self.assertWarnsRegex(
                        UserWarning, 'mutables with `arch param` detected'):
                    _ = mutator.max_choices

                with self.assertWarnsRegex(
                        UserWarning, 'mutables with `arch param` detected'):
                    _ = mutator.min_choices

                with self.assertWarnsRegex(
                        UserWarning, 'mutables with `arch param` detected'):
                    mutator.set_max_choices()

                with self.assertWarnsRegex(
                        UserWarning, 'mutables with `arch param` detected'):
                    mutator.set_min_choices()

                mutator.set_choices(choices)

                x = torch.rand([1, 3, 224, 224])
                y = model(x)
                self.assertEqual(list(y.shape), [1, 32, 114, 114])
