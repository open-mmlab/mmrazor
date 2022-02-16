# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.builder import OPS


def test_common_ops():
    tensor = torch.randn(16, 16, 32, 32)

    # test stride != 1
    identity_cfg = dict(
        type='Identity', in_channels=16, out_channels=16, stride=2)

    op = OPS.build(identity_cfg)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 16

    # test stride == 1
    identity_cfg = dict(
        type='Identity', in_channels=16, out_channels=16, stride=1)

    op = OPS.build(identity_cfg)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

    # test in_channels != out_channels
    identity_cfg = dict(
        type='Identity', in_channels=8, out_channels=16, stride=1)

    op = OPS.build(identity_cfg)

    # test forward
    outputs = op(tensor[:, :8])
    assert outputs.size(1) == 16 and outputs.size(2) == 32


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


def test_mbv2_series():

    tensor = torch.randn(16, 16, 32, 32)

    kernel_sizes = (3, 5, 7)
    expand_factors = (3, 6)
    strides = (1, 2)
    se_cfgs = (None, None)

    for kernel_size in kernel_sizes:
        for expand_factor in expand_factors:
            for stride in strides:
                for se_cfg in se_cfgs:
                    op_cfg = dict(
                        type='MBV2Block',
                        in_channels=16,
                        out_channels=16,
                        kernel_size=kernel_size,
                        expand_factor=expand_factor,
                        se_cfg=se_cfg,
                        stride=stride)

                    op = OPS.build(op_cfg)

                    # test forward
                    outputs = op(tensor)
                    assert outputs.size(1) == 16 and outputs.size(
                        2) == 32 // stride


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
