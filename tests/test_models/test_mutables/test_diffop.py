# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.registry import MODELS

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='torchMaxPool2d', module=nn.MaxPool2d, force=True)
MODELS.register_module(name='torchAvgPool2d', module=nn.AvgPool2d, force=True)


class TestDiffOP(TestCase):

    def test_forward_arch_param(self):
        op_cfg = dict(
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

        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        arch_param = nn.Parameter(torch.randn(len(op_cfg['candidates'])))
        output = op.forward_arch_param(input, arch_param=arch_param)
        assert output is not None

        # test when some element of arch_param is 0
        arch_param = nn.Parameter(torch.ones(op.num_choices))
        output = op.forward_arch_param(input, arch_param=arch_param)
        assert output is not None

    def test_forward_fixed(self):
        op_cfg = dict(
            type='DiffMutableOP',
            candidates=dict(
                torch_conv2d_3x3=dict(
                    type='torchConv2d',
                    kernel_size=3,
                ),
                torch_conv2d_5x5=dict(
                    type='torchConv2d',
                    kernel_size=5,
                ),
                torch_conv2d_7x7=dict(
                    type='torchConv2d',
                    kernel_size=7,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        op.fix_chosen('torch_conv2d_7x7')
        output = op.forward_fixed(input)

        assert output is not None
        assert op.is_fixed is True

    def test_forward(self):
        op_cfg = dict(
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

        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        # test set_forward_args
        arch_param = nn.Parameter(torch.randn(len(op_cfg['candidates'])))
        op.set_forward_args(arch_param=arch_param)
        output = op.forward(input)
        assert output is not None

        # test dump_chosen
        with pytest.raises(AssertionError):
            op.dump_chosen()

        # test forward when is_fixed is True
        op.fix_chosen('torch_conv2d_7x7')
        output = op.forward(input)

    def test_property(self):
        op_cfg = dict(
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

        op = MODELS.build(op_cfg)

        assert len(op.choices) == 3

        # test is_fixed propty
        assert op.is_fixed is False

        # test is_fixed setting
        op.fix_chosen('torch_conv2d_5x5')

        with pytest.raises(AttributeError):
            op.is_fixed = True

        # test fix choice when is_fixed is True
        with pytest.raises(AttributeError):
            op.fix_chosen('torch_conv2d_3x3')

    def test_module_kwargs(self):
        op_cfg = dict(
            type='DiffMutableOP',
            candidates=dict(
                torch_conv2d_3x3=dict(
                    type='torchConv2d',
                    kernel_size=3,
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                ),
                torch_conv2d_5x5=dict(
                    type='torchConv2d',
                    kernel_size=5,
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                ),
                torch_conv2d_7x7=dict(
                    type='torchConv2d',
                    kernel_size=7,
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                ),
                torch_maxpool_3x3=dict(
                    type='torchMaxPool2d',
                    kernel_size=3,
                    stride=1,
                ),
                torch_avgpool_3x3=dict(
                    type='torchAvgPool2d',
                    kernel_size=3,
                    stride=1,
                ),
            ),
        )
        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        op.fix_chosen('torch_avgpool_3x3')
        output = op.forward(input)
        assert output is not None
