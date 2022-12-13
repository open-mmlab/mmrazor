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

try:
    from mmpose.models import PoseDataPreProcessor
    from mmpose.structures import PoseDataSample
except ImportError:
    PoseDataPreProcessor = get_placeholder('mmpose')
    PoseDataSample = get_placeholder('mmpose')


def demo_mmseg_inputs(model, input_shape, for_training=False):


    mm_inputs = _demo_mmpose_inputs(input_shape=input_shape)

    # convert to cuda Tensor if applicabled
    # if torch.cuda.is_available():
    #     segmentor = segmentor.cuda()

    # check data preprocessor
    if not hasattr(model,
                   'data_preprocessor') or model.data_preprocessor is None:
        model.data_preprocessor = PoseDataPreProcessor()

    mm_inputs = model.data_preprocessor(mm_inputs, for_training)

    return mm_inputs


def _demo_mmpose_inputs(input_shape=(1, 3, 192, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    imgs = torch.randn(*input_shape)

    rng = np.random.RandomState(0)

    data_samples = []
    for i in range(N):
        img_meta = {
            'id': i,
            'img_id': i,
            'img_shape': (H, W),
            'input_size': (H, W),
            'filename': '<demo>.png',
            'flip': False,
            'flip_direction': None,
            'flip_indices': list(range(17))
        }

        np.random.shuffle(img_meta['flip_indices'])
        data_sample = PoseDataSample(metainfo=img_meta)

        gt_instances = InstanceData()
        gt_instance_labels = InstanceData()


        bboxes = _rand_bboxes(rng, 1, W, H)
        bbox_centers, bbox_scales = bbox_xyxy2cs(bboxes)

        keypoints = _rand_keypoints(rng, bboxes, 17)
        keypoints_visible = np.ones((1, 17),
                                dtype=np.float32)

        keypoint_weights = keypoints_visible.copy()

        gt_instances.bboxes = bboxes
        gt_instances.bbox_centers = bbox_centers
        gt_instances.bbox_scales = bbox_scales
        gt_instances.bbox_scores = np.ones((1, ), dtype=np.float32)
        gt_instances.keypoints = keypoints
        gt_instances.keypoints_visible = keypoints_visible

        gt_instance_labels.keypoint_weights = torch.FloatTensor(
            keypoint_weights)

        gt_fields = PixelData()
        # generate single-level heatmaps
        heatmaps = rng.rand(17, 64, 48)
        gt_fields.heatmaps = torch.FloatTensor(heatmaps)
        data_sample.gt_fields = gt_fields

        data_sample.gt_instances = gt_instances
        data_sample.gt_instance_labels = gt_instance_labels

        data_samples.append(data_sample)

    mm_inputs = {
        'inputs': torch.FloatTensor(imgs),
        'data_samples': data_samples
        }

    return data_sample

def _rand_keypoints(rng, bboxes, num_keypoints):
    n = bboxes.shape[0]
    relative_pos = rng.rand(n, num_keypoints, 2)
    keypoints = relative_pos * bboxes[:, None, :2] + (
        1 - relative_pos) * bboxes[:, None, 2:4]

    return keypoints


def _rand_simcc_label(rng, num_instances, num_keypoints, len_feats):
    simcc_label = rng.rand(num_instances, num_keypoints, int(len_feats))
    return simcc_label


def _rand_bboxes(rng, num_instances, img_w, img_h):
    cx, cy, bw, bh = rng.rand(num_instances, 4).T

    tl_x = ((cx * img_w) - (img_w * bw / 2)).clip(0, img_w)
    tl_y = ((cy * img_h) - (img_h * bh / 2)).clip(0, img_h)
    br_x = ((cx * img_w) + (img_w * bw / 2)).clip(0, img_w)
    br_y = ((cy * img_h) + (img_h * bh / 2)).clip(0, img_h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return 

# a = _demo_mmpose_inputs
# print(a.gt_intances)