# Copyright (c) OpenMMLab. All rights reserved.
"""Include functions to generate mmpose demo inputs.

Modified from mmpose.
"""

import torch
from mmpose.models.heads import (CPMHead, DSNTHead, HeatmapHead,
                                 IntegralRegressionHead, MSPNHead,
                                 RegressionHead, RLEHead, SimCCHead,
                                 ViPNASHead)
from mmpose.testing._utils import get_packed_inputs

from mmrazor.utils import get_placeholder

try:
    from mmpose.models import PoseDataPreProcessor
    from mmpose.structures import PoseDataSample
except ImportError:
    PoseDataPreProcessor = get_placeholder('mmpose')
    PoseDataSample = get_placeholder('mmpose')


def demo_mmpose_inputs(model, for_training=False, batch_size=1):
    input_shape = (
        1,
        3,
    ) + model.head.decoder.input_size
    imgs = torch.randn(*input_shape)

    batch_data_samples = []
    from mmpose.models.heads import RTMHead
    if isinstance(model.head, HeatmapHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.out_channels,
            heatmap_size=model.head.decoder.heatmap_size[::-1])['data_samples']
    elif isinstance(model.head, MSPNHead):
        batch_data_samples = get_packed_inputs(
            batch_size=batch_size,
            num_instances=1,
            num_keypoints=model.head.out_channels,
            heatmap_size=model.head.decoder.heatmap_size,
            with_heatmap=True,
            with_reg_label=False,
            num_levels=model.head.num_stages *
            model.head.num_units)['data_samples']
    elif isinstance(model.head, CPMHead):
        batch_data_samples = get_packed_inputs(
            batch_size=batch_size,
            num_instances=1,
            num_keypoints=model.head.out_channels,
            heatmap_size=model.head.decoder.heatmap_size[::-1],
            with_heatmap=True,
            with_reg_label=False)['data_samples']

    elif isinstance(model.head, SimCCHead):
        # bug
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.out_channels,
            simcc_split_ratio=model.head.decoder.simcc_split_ratio,
            input_size=model.head.decoder.input_size,
            with_simcc_label=True)['data_samples']

    elif isinstance(model.head, ViPNASHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.out_channels,
        )['data_samples']

    elif isinstance(model.head, DSNTHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.num_joints,
            with_reg_label=True)['data_samples']

    elif isinstance(model.head, IntegralRegressionHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.num_joints,
            with_reg_label=True)['data_samples']

    elif isinstance(model.head, RegressionHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.num_joints,
            with_reg_label=True)['data_samples']

    elif isinstance(model.head, RLEHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.num_joints,
            with_reg_label=True)['data_samples']

    elif isinstance(model.head, RTMHead):
        batch_data_samples = get_packed_inputs(
            batch_size,
            num_keypoints=model.head.out_channels,
            simcc_split_ratio=model.head.decoder.simcc_split_ratio,
            input_size=model.head.decoder.input_size,
            with_simcc_label=True)['data_samples']

    else:
        raise AssertionError(f'Head Type {type(model.head)} is Not Predefined')

    mm_inputs = {
        'inputs': torch.FloatTensor(imgs),
        'data_samples': batch_data_samples
    }

    # check data preprocessor
    if not hasattr(model,
                   'data_preprocessor') or model.data_preprocessor is None:
        model.data_preprocessor = PoseDataPreProcessor()

    mm_inputs = model.data_preprocessor(mm_inputs, for_training)

    return mm_inputs
