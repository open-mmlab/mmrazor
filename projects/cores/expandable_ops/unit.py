import copy

import torch
import torch.nn as nn

from mmrazor.models.mutables import (L1MutableChannelUnit,
                                     MutableChannelContainer)
from mmrazor.models.mutators import ChannelMutator
from .ops import (ExpandableBatchNorm2d, ExpandableConv2d, ExpandableMixin,
                  ExpandLinear)


def expand_static_model(model: nn.Module, divisor):
    """Expand the channels of a model.

    Args:
        model (nn.Module): the model to be expanded.
        divisor (_type_): the divisor to make the channels divisible.

    Returns:
        nn.Module: an expanded model.
    """
    from projects.cores.expandable_ops.unit import ExpandableUnit, expand_model
    state_dict = model.state_dict()
    mutator = ChannelMutator[ExpandableUnit](channel_unit_cfg=ExpandableUnit)
    mutator.prepare_from_supernet(model)
    model.load_state_dict(state_dict)
    for unit in mutator.mutable_units:
        num = unit.current_choice
        if num % divisor == 0:
            continue
        else:
            num = (num // divisor + 1) * divisor
            num = max(num, unit.num_channels)
            unit.expand_to(num)
    expand_model(model, zero=True)

    mutator = ChannelMutator[ExpandableUnit](channel_unit_cfg=ExpandableUnit)
    mutator.prepare_from_supernet(copy.deepcopy(model))
    structure = mutator.choice_template
    return structure


def expand_model(model: nn.Module, zero=False) -> None:

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


class ExpandableUnit(L1MutableChannelUnit):

    def prepare_for_pruning(self, model: nn.Module):
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: ExpandableConv2d,
                nn.BatchNorm2d: ExpandableBatchNorm2d,
                nn.Linear: ExpandLinear,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def expand(self, num):
        expand_mask = self.mutable_channel.mask.new_zeros([num])
        mask = torch.cat([self.mutable_channel.mask, expand_mask])
        self.mutable_channel.mask = mask

    def expand_to(self, num):
        self.expand(num - self.num_channels)
