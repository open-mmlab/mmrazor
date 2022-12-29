# Copyright (c) OpenMMLab. All rights reserved.
"""Include functions to generate mmpose demo inputs.

Modified from mmpose.
"""

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData, PixelData
from torch import nn

from mmrazor.utils import get_placeholder

from mmpose.structures.bbox import bbox_xyxy2cs
from mmpose.testing._utils import get_packed_inputs

# heatmap heads
from mmpose.models.heads import HeatmapHead, MSPNHead, CPMHead, SimCCHead, ViPNASHead
# regression heads
from mmpose.models.heads import DSNTHead, IntegralRegressionHead, RegressionHead, RLEHead


try:
    from mmpose.models import PoseDataPreProcessor
    from mmpose.structures import PoseDataSample
except ImportError:
    PoseDataPreProcessor = get_placeholder('mmpose')
    PoseDataSample = get_placeholder('mmpose')


def demo_mmpose_inputs(model, for_training=False, batch_size=1):

    input_shape = (1,3,) + model.head.decoder.input_size
    imgs = torch.randn(*input_shape)

    batch_data_samples = []

    if isinstance(model.head, HeatmapHead):
        batch_data_samples = [inputs['data_sample'] for inputs in get_packed_inputs(
            batch_size, 
            num_keypoints=model.head.out_channels,
            heatmap_size=model.head.decoder.heatmap_size[::-1])
            ]
    elif isinstance(model.head, MSPNHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size=batch_size,
                num_instances=1,
                num_keypoints=model.head.out_channels,
                heatmap_size=model.head.decoder.heatmap_size,
                with_heatmap=True,
                with_reg_label=False,
                num_levels=model.head.num_stages*model.head.num_units)
        ]
    elif isinstance(model.head, CPMHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size=batch_size,
                num_instances=1,
                num_keypoints=model.head.out_channels,
                heatmap_size=model.head.decoder.heatmap_size[::-1],
                with_heatmap=True,
                with_reg_label=False)
        ]
    elif isinstance(model.head, SimCCHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.out_channels,
                simcc_split_ratio=model.head.decoder.simcc_split_ratio,
                input_size=model.head.decoder.input_size,
                with_simcc_label=True)
        ]
    elif isinstance(model.head, ViPNASHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.out_channels,)
        ]
    elif isinstance(model.head, DSNTHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.num_joints,
                with_reg_label=True)
        ]
    elif isinstance(model.head, IntegralRegressionHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.num_joints,
                with_reg_label=True)
        ]
    elif isinstance(model.head, RegressionHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.num_joints,
                with_reg_label=True)
        ]
    elif isinstance(model.head, RLEHead):
        batch_data_samples = [
            inputs['data_sample'] for inputs in get_packed_inputs(
                batch_size,
                num_keypoints=model.head.num_joints,
                with_reg_label=True)
        ]        
    else:
        raise AssertionError('Head Type is Not Predefined')


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


# def _demo_mmpose_inputs(input_shape=(1, 3, 192, 256)):
#     """Create a superset of inputs needed to run test or train batches.

#     Args:
#         input_shape (tuple):
#             input batch dimensions

#         num_classes (int):
#             number of semantic classes
#     """
#     imgs = torch.randn(*input_shape)

#     data = get_packed_inputs(2, img_shape=[256, 198])
#     data_samples = [d['data_sample'] for d in data]

#     mm_inputs = {
#         'inputs': torch.FloatTensor(imgs),
#         'data_samples': data_samples
#         }

#     return mm_inputs
