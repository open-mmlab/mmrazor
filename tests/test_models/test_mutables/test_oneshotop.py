# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

import mmrazor.models  # noqa:F401
from mmrazor.registry import MODELS


class TestMutables(TestCase):

    def test_oneshotmutableop(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        op_cfg = dict(
            type='OneShotMutableOP',
            candidates=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        # test forward all
        output = op.forward_all(input)
        assert output is not None

        # test random choice
        assert op.sample_choice() in [
            'shuffle_3x3', 'shuffle_5x5', 'shuffle_7x7', 'shuffle_xception'
        ]

        # test unfixed mode
        op.current_choice = 'shuffle_3x3'
        output1 = op.forward(input)

        op.current_choice = 'shuffle_7x7'
        output2 = op.forward(input)

        assert not output1.equal(output2)

        assert op.is_fixed is False
        assert len(op.choices) == 4
        assert op.num_choices == 4

        # compare set_forward_args with forward with choice
        op.current_choice = 'shuffle_5x5'
        output1 = op.forward(input)
        output2 = op.forward_choice(input, choice='shuffle_5x5')
        assert output1.equal(output2)

        # test fixed mode
        op.fix_chosen('shuffle_3x3')
        assert op.is_fixed is True
        assert len(op.choices) == 1
        assert op.num_choices == 1

        output = op.forward(input)
        assert output.shape[1] == 32

        with pytest.raises(AttributeError):
            op.is_fixed = True

        with pytest.raises(AttributeError):
            op.fix_chosen('shuffle_3x3')

    def test_oneshotprobop(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        op_cfg = dict(
            type='OneShotProbMutableOP',
            choice_probs=[0.1, 0.2, 0.3, 0.4],
            candidates=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        op = MODELS.build(op_cfg)

        input = torch.randn(4, 32, 64, 64)

        # test forward choice with None
        with pytest.raises(AssertionError):
            output = op.forward_choice(input, choice=None)

        # test forward all
        output = op.forward_all(input)
        assert output.shape[1] == 32

        # test random choice
        assert op.sample_choice() in [
            'shuffle_3x3', 'shuffle_5x5', 'shuffle_7x7', 'shuffle_xception'
        ]
        assert 1 - sum(op.choice_probs) < 0.00001

        # test unfixed mode
        op.current_choice = 'shuffle_3x3'
        output = op.forward(input)

        assert output.shape[1] == 32

        op.current_choice = 'shuffle_7x7'
        output = op.forward(input)
        assert output.shape[1] == 32

        assert op.is_fixed is False
        assert len(op.choices) == 4
        assert op.num_choices == 4

        # test fixed mode
        op.fix_chosen('shuffle_3x3')
        assert op.is_fixed is True
        assert len(op.choices) == 1
        assert op.num_choices == 1

        output = op.forward(input)
        assert output.shape[1] == 32

        with pytest.raises(AttributeError):
            op.is_fixed = True

    def test_forward_choice(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        op_cfg = dict(
            type='OneShotMutableOP',
            candidates=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        assert op.forward_choice(input, choice='shuffle_3x3') is not None

    def test_fix_chosen(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        op_cfg = dict(
            type='OneShotMutableOP',
            candidates=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ),
            module_kwargs=dict(in_channels=32, out_channels=32, stride=1))

        op = MODELS.build(op_cfg)

        with pytest.raises(AttributeError):
            op.fix_chosen('shuffle_xception')
            op.fix_chosen('ShuffleBlock')

    def test_build_ops(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        op_cfg = dict(
            type='OneShotMutableOP',
            candidates=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=3,
                    in_channels=32,
                    out_channels=32),
                shuffle_5x5=dict(
                    type='ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=5,
                    in_channels=32,
                    out_channels=32),
                shuffle_7x7=dict(
                    type='ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=7,
                    in_channels=32,
                    out_channels=32),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                    in_channels=32,
                    out_channels=32),
            ),
        )
        op = MODELS.build(op_cfg)
        input = torch.randn(4, 32, 64, 64)

        output = op.forward_all(input)
        assert output is not None

    def test_candidates(self):

        candidates = nn.ModuleDict({
            'conv3x3': nn.Conv2d(32, 32, 3, 1, 1),
            'conv5x5': nn.Conv2d(32, 32, 5, 1, 2),
            'conv7x7': nn.Conv2d(32, 32, 7, 1, 3),
            'maxpool3x3': nn.MaxPool2d(3, 1, 1),
            'avgpool3x3': nn.AvgPool2d(3, 1, 1),
        })

        op_cfg = dict(type='OneShotMutableOP', candidates=candidates)

        op = MODELS.build(op_cfg)

        input = torch.randn(4, 32, 64, 64)

        output = op.forward_all(input)
        assert output is not None
