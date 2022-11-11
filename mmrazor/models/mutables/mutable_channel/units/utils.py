# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit
from mmrazor.utils import demo_inputs


def assert_model_is_changed(tensors1, tensors2):
    shape1 = get_shape(tensors1, only_length=True)
    shape2 = get_shape(tensors2, only_length=True)
    assert shape1 == shape2, f'{shape1}!={shape2}'


def get_shape(tensor, only_length=False):
    if isinstance(tensor, torch.Tensor):
        if only_length:
            return len(tensor.shape)
        else:
            return tensor.shape
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        shapes = []
        for x in tensor:
            shapes.append(get_shape(x, only_length))
        return shapes
    elif isinstance(tensor, dict):
        shapes = {}
        for key in tensor:
            shapes[key] = get_shape(tensor[key], only_length)
        return shapes
    else:
        raise NotImplementedError(
            f'unsuppored type{type(tensor)} to get shape of tensors.')


def forward_units(model, try_units: List[SequentialMutableChannelUnit],
                  units: List[SequentialMutableChannelUnit], template_output):
    model.eval()
    for unit in units:
        unit.current_choice = 1.0
    for unit in try_units:
        unit.current_choice = min(max(0.1, unit.sample_choice()), 0.9)
    inputs = demo_inputs(model, [1, 3, 224, 224])
    if isinstance(inputs, dict):
        inputs['mode'] = 'loss'
        tensors = model(**inputs)
    else:
        tensors = model(inputs)
    assert_model_is_changed(template_output, tensors)


def find_mutable(model, try_units, units, template_tensors):
    if len(try_units) == 0:
        return []
    try:
        forward_units(model, try_units, units, template_tensors)
        return try_units
    except Exception:
        if len(try_units) == 1:
            return []
        else:
            num = len(try_units)
            return find_mutable(model, try_units[:num // 2], units,
                                template_tensors) + find_mutable(
                                    model, try_units[num // 2:], units,
                                    template_tensors)
