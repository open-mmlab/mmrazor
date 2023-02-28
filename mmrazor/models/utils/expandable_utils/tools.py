# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch.nn as nn

from mmrazor.models.mutators import ChannelMutator
from .ops import ExpandableMixin
from .unit import ExpandableUnit


def to_expandable_model(model: nn.Module) -> ChannelMutator[ExpandableUnit]:
    """Convert a static model to an expandable model."""
    state_dict = model.state_dict()
    mutator = ChannelMutator[ExpandableUnit](
        channel_unit_cfg=ExpandableUnit,
        parse_cfg=dict(
            _scope_='mmrazor',
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='FxTracer'),
    )
    mutator.prepare_from_supernet(model)
    model.load_state_dict(state_dict)
    return mutator


def expand_expandable_dynamic_model(model: nn.Module, zero=False) -> nn.Module:
    """Expand a expandable model and return a expanded static model.

    Args:
        model (nn.Module): The model to be expanded.
        zero (bool, optional): Whether to zero expanded weight. Defaults to
            False.
    """

    def traverse_children(module: nn.Module) -> None:
        for name, mutable in module.items():
            if isinstance(mutable, ExpandableMixin):
                module[name] = mutable.expand(zero=zero)
            if hasattr(mutable, '_modules'):
                traverse_children(mutable._modules)

    if isinstance(model, ExpandableMixin):
        raise RuntimeError('Root model can not be dynamic op.')

    if hasattr(model, '_modules'):
        traverse_children(model._modules)
    return model


def expand_static_model(model: nn.Module, structure: Dict, zero_weight=True):
    """Expand the channels of a model.

    Args:
        model (nn.Module): the model to be expanded.
        structure (Dict): the channel structure for the model.
        divisor (_type_): the divisor to make the channels divisible.
    """
    mutator = to_expandable_model(model)
    for key, value in structure.items():
        mutator._name2unit[key].expand_to(value)
    expand_expandable_dynamic_model(model, zero=zero_weight)
    return model


def make_channel_divisible(model: nn.Module, divisor, zero_weight=True):
    """Expand the channels of a model and return the new divisible channel
    structure.

    Args:
        model (nn.Module): the model to be expanded.
        divisor (_type_): the divisor to make the channels divisible.
    """
    # to sta
    mutator = to_expandable_model(model)

    structure = mutator.choice_template
    for key, num in structure.items():
        unit = mutator._name2unit[key]
        if num % divisor == 0:
            continue
        else:
            num = (num // divisor + 1) * divisor
            num = max(num, unit.num_channels)
            unit.expand_to(num)

    model = expand_expandable_dynamic_model(model, zero=zero_weight)
    mutator = to_expandable_model(copy.deepcopy(model))

    return mutator.choice_template
