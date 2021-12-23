# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.builder import OPS


def test_shuffle_series():

    tensor = torch.randn(16, 16, 32, 32)

    # test ShuffleBlock_7x7
    shuffle_block_7x7 = dict(
        type='ShuffleBlock',
        in_channels=16,
        out_channels=16,
        kernel_size=7,
        stride=1)

    op = OPS.build(shuffle_block_7x7)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test ShuffleBlock_5x5
    shuffle_block_5x5 = dict(
        type='ShuffleBlock',
        in_channels=16,
        out_channels=16,
        kernel_size=5,
        stride=1)

    op = OPS.build(shuffle_block_5x5)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test ShuffleBlock_3x3
    shuffle_block_3x3 = dict(
        type='ShuffleBlock',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        stride=1)

    op = OPS.build(shuffle_block_3x3)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test ShuffleXception
    shuffle_xception = dict(
        type='ShuffleXception', in_channels=16, out_channels=16, stride=1)

    op = OPS.build(shuffle_xception)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32


def test_darts_series():

    tensor = torch.randn(16, 16, 32, 32)

    # test avg pool bn
    avg_pool_bn = dict(
        type='DartsPoolBN',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        pool_type='avg',
        stride=1)

    op = OPS.build(avg_pool_bn)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test max pool bn
    max_pool_bn = dict(
        type='DartsPoolBN',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        pool_type='max',
        stride=1)

    op = OPS.build(max_pool_bn)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test DartsSepConv
    sep_conv = dict(
        type='DartsSepConv',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        stride=1)

    op = OPS.build(sep_conv)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test DartsSepConv
    sep_conv = dict(
        type='DartsSepConv',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        stride=1)

    op = OPS.build(sep_conv)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test DartsDilConv
    dil_conv = dict(
        type='DartsDilConv',
        in_channels=16,
        out_channels=16,
        kernel_size=3,
        stride=1)

    op = OPS.build(dil_conv)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test DartsSkipConnect
    skip_connect = dict(
        type='DartsSkipConnect', in_channels=16, out_channels=16, stride=1)

    op = OPS.build(skip_connect)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32
