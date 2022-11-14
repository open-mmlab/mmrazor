# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..placeholder import get_placeholder

try:
    from mmcls.models import ImageClassifier
    from mmcls.structures import ClsDataSample
except ImportError:
    ImageClassifier = get_placeholder('mmcls')
    ClsDataSample = get_placeholder('mmcls')


def mmcls_demo_input(model: ImageClassifier, input_shape: tuple):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    x = torch.rand(input_shape)
    mm_inputs = {
        'inputs':
        x,
        'data_samples':
        [ClsDataSample().set_gt_label(1) for _ in range(input_shape[0])],
    }
    mm_inputs['mode'] = 'tensor'
    return mm_inputs
