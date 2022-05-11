# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmrazor.models.ops import *
from mmrazor.registry import MODELS

#TODO complete
def test_shuffle_block():

    tensor = torch.randn(16, 16, 32, 32)

    # test ShuffleBlock_7x7
    shuffle_block_7x7 = dict(
        type='ShuffleBlock',
        in_channels=16,
        out_channels=16,
        kernel_size=7,
        stride=1)

    op = MODELS.build(shuffle_block_7x7)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32

def test_shuffle_xception():

    tensor = torch.randn(16, 16, 32, 32)

    # test ShuffleXception
    shuffle_xception = dict(
        type='ShuffleXception', in_channels=16, out_channels=16, stride=1)

    op = MODELS.build(shuffle_xception)

    # test forward
    outputs = op(tensor)
    assert outputs.size(1) == 16 and outputs.size(2) == 32