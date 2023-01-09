# Copyright (c) OpenMMLab. All rights reserved.
"""Include functions to generate mmsegementation demo inputs.

Modified from mmseg.
"""
import torch
from mmengine.structures import PixelData
from torch import nn

from mmrazor.utils import get_placeholder

try:
    from mmseg.models import SegDataPreProcessor
    from mmseg.structures import SegDataSample
except ImportError:
    SegDataPreProcessor = get_placeholder('mmseg')
    SegDataSample = get_placeholder('mmseg')


def demo_mmseg_inputs(segmentor, input_shape, for_training=False):

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    # batch_size=2 for BatchNorm
    mm_inputs = _demo_mmseg_inputs(
        num_classes=num_classes, input_shape=input_shape)

    # convert to cuda Tensor if applicabled
    # if torch.cuda.is_available():
    #     segmentor = segmentor.cuda()

    # check data preprocessor
    if not hasattr(segmentor,
                   'data_preprocessor') or segmentor.data_preprocessor is None:
        segmentor.data_preprocessor = SegDataPreProcessor()

    mm_inputs = segmentor.data_preprocessor(mm_inputs, for_training)

    return mm_inputs


def _demo_mmseg_inputs(input_shape=(1, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    imgs = torch.randn(*input_shape)
    segs = torch.randint(
        low=0, high=num_classes - 1, size=(N, H, W), dtype=torch.long)

    img_metas = [{
        'img_shape': (H, W),
        'ori_shape': (H, W),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    data_samples = [
        SegDataSample(
            gt_sem_seg=PixelData(data=segs[i]), metainfo=img_metas[i])
        for i in range(N)
    ]

    mm_inputs = {
        'inputs': torch.FloatTensor(imgs),
        'data_samples': data_samples
    }

    return mm_inputs
