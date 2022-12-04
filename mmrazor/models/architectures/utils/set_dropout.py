# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ..dynamic_ops.bricks import DynamicSequential
from ..ops.mobilenet_series import MBBlock


def set_dropout(layers, dropout_stages: List[int],
                drop_path_rate: float) -> None:
    """Set dropout for layers in MobileNet series network.

    Args:
        layers: Layers in MobileNet-style networks.
        dropout_stages (List[int]): Stages to be set dropout.
        drop_path_rate (float): Drop path rate for layers.
    """
    visited_block_nums = 0
    total_block_nums = sum(len(layer) for layer in layers) + 1

    for idx, layer in enumerate(layers, start=1):
        assert isinstance(layer, DynamicSequential)
        visited_block_nums += len(layer)
        if idx not in dropout_stages:
            continue

        for block_idx, block in enumerate(layer):
            if isinstance(block, MBBlock) and hasattr(block, 'drop_path_rate'):
                ratio = (visited_block_nums - len(layer) +
                         block_idx) / total_block_nums
                block.drop_path_rate = drop_path_rate * ratio
