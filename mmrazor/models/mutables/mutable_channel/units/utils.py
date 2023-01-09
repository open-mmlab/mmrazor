# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit
from mmrazor.utils import print_log


def assert_model_is_changed(tensors1, tensors2):
    """Return if the tensors has the same shape (length)."""
    shape1 = get_shape(tensors1, only_length=True)
    shape2 = get_shape(tensors2, only_length=True)
    assert shape1 == shape2, f'{shape1}!={shape2}'


def get_shape(tensor, only_length=False):
    """Get the shape of a tensor list/tuple/dict.

    Args:
        tensor (Union[List,Tuple,Dict,Tensor]): input tensors.
        only_length (bool, optional): If only return the length of the tensors.
            Defaults to False.
    """
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
                  units: List[SequentialMutableChannelUnit], demo_input,
                  template_output):
    """Forward a model with MutableChannelUnits and assert if the result
    changed."""
    model.eval()
    for unit in units:
        unit.current_choice = 1.0
    for unit in try_units:
        unit.current_choice = min(max(0.1, unit.sample_choice()), 0.9)
    if isinstance(demo_input, dict):
        tensors = model(**demo_input)
    else:
        tensors = model(demo_input)
    assert_model_is_changed(template_output, tensors)


def find_mutable(model, try_units, units, demo_input, template_tensors):
    """Find really mutable MutableChannelUnits in some MutableChannelUnits."""
    if len(try_units) == 0:
        return []
    try:
        forward_units(model, try_units, units, demo_input, template_tensors)
        return try_units
    except Exception:
        if len(try_units) == 1:
            print_log(f'Find an unmutable unit {try_units[0]}', level='debug')
            return []
        else:
            num = len(try_units)
            return find_mutable(model, try_units[:num // 2], units, demo_input,
                                template_tensors) + find_mutable(
                                    model, try_units[num // 2:], units,
                                    demo_input, template_tensors)
