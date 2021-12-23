# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.builder import MUTABLES


def test_one_shot_op():
    oneshot_choice_op = dict(
        type='OneShotOP',
        space_id='test',
        num_chosen=1,
        choices=dict(
            shuffle_3x3=dict(type='ShuffleBlock', kernel_size=3),
            shuffle_5x5=dict(type='ShuffleBlock', kernel_size=5),
            shuffle_7x7=dict(type='ShuffleBlock', kernel_size=7),
            shuffle_xception=dict(type='ShuffleXception'),
        ),
        choice_args=dict(in_channels=16, out_channels=16, stride=1))

    model = MUTABLES.build(oneshot_choice_op)

    tensor = torch.randn(16, 16, 32, 32)

    # test forward
    outputs = model(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32


def test_differentiable_op():
    oneshot_choice_op = dict(
        type='DifferentiableOP',
        space_id='test',
        num_chosen=1,
        with_arch_param=True,
        choices=dict(
            zero=dict(type='DartsZero'),
            skip_connect=dict(type='DartsSkipConnect'),
            dil_conv_3x3=dict(type='DartsDilConv', kernel_size=3),
            dil_conv_5x5=dict(type='DartsDilConv', kernel_size=5),
            sep_conv_3x3=dict(type='DartsSepConv', kernel_size=3),
            sep_conv_5x5=dict(type='DartsSepConv', kernel_size=5),
        ),
        choice_args=dict(in_channels=16, out_channels=16, stride=2))

    model = MUTABLES.build(oneshot_choice_op)
    arch_param = model.build_arch_param()

    tensor = torch.randn(16, 16, 32, 32)

    # test forward
    outputs = model(tensor, arch_param)
    assert outputs.size(1) == 16 and outputs.size(2) == 16
