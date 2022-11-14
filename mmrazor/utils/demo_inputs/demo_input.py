# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModel

from ..placeholder import get_placeholder
from .mmcls_demo_input import mmcls_demo_input

try:
    from mmdet.models import BaseDetector
except Exception:
    BaseDetector = get_placeholder('mmdet')

try:
    from mmcls.models import ImageClassifier
except Exception:
    ImageClassifier = get_placeholder('mmcls')

try:
    from mmseg.models import BaseSegmentor
except Exception:
    BaseSegmentor = get_placeholder('mmseg')


def default_mm_concrete_args(model, input_shape):
    x = torch.rand(input_shape)
    return {'inputs': x, 'mode': 'tensor'}


def default_concrete_args(model, input_shape):
    x = torch.rand(input_shape)
    return x


def seg_concrete_args(model, input_shape):
    assert isinstance(model, BaseSegmentor)
    from .mmseg_demo_input import demo_mmseg_inputs
    data = demo_mmseg_inputs(model, input_shape)
    data['mode'] = 'tensor'
    return data


def det_concrete_args(model, input_shape):
    assert isinstance(model, BaseDetector)
    from mmdet.testing._utils import demo_mm_inputs
    data = demo_mm_inputs(1, [input_shape[1:]])
    data = model.data_preprocessor(data, False)
    data['mode'] = 'tensor'
    return data


default_concrete_args_fun = {
    BaseDetector: det_concrete_args,
    ImageClassifier: mmcls_demo_input,
    BaseSegmentor: seg_concrete_args,
    BaseModel: default_mm_concrete_args,
    nn.Module: default_concrete_args
}


def demo_inputs(model, input_shape):
    for module_type, concrete_args_fun in default_concrete_args_fun.items(  # noqa
    ):  # noqa
        if isinstance(model, module_type):
            return concrete_args_fun(model, input_shape)
    # default
    x = torch.rand(input_shape)
    return x
