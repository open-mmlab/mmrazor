# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines MutableChannelGroup."""
import abc
from typing import Dict, List, Type, TypeVar, Union

import torch.nn as nn
from mmengine.model import BaseModule

from mmrazor.models.architectures.dynamic_ops.bricks import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel.base_mutable_channel import \
    BaseMutableChannel
from ..mutable_channel_container import MutableChannelContainer
from .channel_group import Channel, ChannelGroup


class MutableChannelGroup(ChannelGroup, BaseModule):
    """MutableChannelGroup inherits from ChannelGroup, which manages channels
    with channel-dependency.

    Compared with ChannelGroup, MutableChannelGroup defines the core
    interfaces for pruning. By inheriting MutableChannelGroup, we can implement
    a variant pruning algorithm.

    Basic Property

        name
        is_prunable

    Important interfaces during different stages:

    # Before pruning
        prepare_model
        prepare_for_pruning

    # Pruning stage
        current_choice
        sample_choice

    # After pruning
        fix_chosen
    """

    def __init__(self, num_channels: int) -> None:
        """
        Args:
            num_channels (int): dimension of the channels that this
            MutableChannelGroup manages.
        """
        super().__init__(num_channels)
        BaseModule.__init__(self)

    # basic property

    @property
    def name(self) -> str:
        """str: name of the group"""
        first_module = self.output_related[0] if len(
            self.output_related) > 0 else self.input_related[0]
        name = f'{first_module.name}_{first_module.index}_'
        name += f'out_{len(self.output_related)}_in_{len(self.input_related)}'
        return name

    @property
    def is_prunable(self) -> bool:
        """If the channel-group is prunable."""

        def traverse(channels: List[Channel]):
            has_dynamic_op = False
            all_channel_prunable = True
            for channel in channels:
                if channel.node.is_prunable is False:
                    all_channel_prunable = False
                    break
                if isinstance(channel.module, DynamicChannelMixin):
                    has_dynamic_op = True
            return has_dynamic_op, all_channel_prunable

        input_has_dynamic_op, input_all_prunable = traverse(self.input_related)
        output_has_dynamic_op, output_all_prunable = traverse(
            self.output_related)

        return len(self.output_related) > 0 \
            and len(self.input_related) > 0 \
            and input_has_dynamic_op \
            and input_all_prunable \
            and output_has_dynamic_op \
            and output_all_prunable

    # before pruning: prepare a model

    @abc.abstractmethod
    def prepare_for_pruning(self, model):
        """Post process after parse groups.

        For example, we need to register mutables to dynamic-ops.
        """
        raise not NotImplementedError

    # pruning: choice-related

    @property
    def current_choice(self):
        """Choice of this group."""
        raise NotImplementedError()

    @current_choice.setter
    def current_choice(self, choice) -> None:
        """setter of current_choice."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_choice(self):
        """Randomly sample a valid choice and return."""
        raise NotImplementedError()

    def config_template(self, with_info=False):
        if with_info:
            return {'info': self._info_dict}
        else:
            return {}

    # after pruning

    def fix_chosen(self, choice=None):
        """Make the channels in this group fixed."""
        if choice is not None:
            self.current_choice = choice

    # tools

    def _info_dict(self):
        info = {
            'num_channels': self.num_channels,
            'choice': self.current_choice,
            'prunable': self.is_prunable,
            'input_layers': self.input_related,
            'out_related': self.output_related
        }
        return info

    def _get_int_choice(self, choice: Union[int, float]) -> int:
        """Convert ratio of channels to number of channels."""
        if isinstance(choice, float):
            choice = max(1, int(self.num_channels * choice))
        assert 0 < choice <= self.num_channels, f'{choice}'
        return choice

    def _replace_with_dynamic_ops(
            self, model: nn.Module,
            dynamicop_map: Dict[Type[nn.Module], Type[DynamicChannelMixin]]):
        """Replace torch modules with dynamic-ops."""

        def replace_op(model: nn.Module, name: str, module: nn.Module):
            names = name.split('.')
            for sub_name in names[:-1]:
                model = getattr(model, sub_name)

            setattr(model, names[-1], module)

        def get_module(model, name):
            names = name.split('.')
            for sub_name in names:
                model = getattr(model, sub_name)
            return model

        for channel in self.input_related + self.output_related:
            if isinstance(channel.module, nn.Module):
                module = get_module(model, channel.module_name)
                if type(module) in dynamicop_map:
                    new_module = dynamicop_map[type(module)].convert_from(
                        module)
                    replace_op(model, channel.module_name, new_module)
                    channel.module = new_module
                else:
                    channel.module = module

    @staticmethod
    def _register_mask_container(
            model: nn.Module, container_class: Type[MutableChannelContainer]):
        """register channel container for dynamic ops."""
        for module in model.modules():
            if isinstance(module, DynamicChannelMixin):
                if module.get_mutable_attr('in_channels') is None:
                    in_channels = 0
                    if isinstance(module, nn.Conv2d):
                        in_channels = module.in_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        in_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        in_channels = module.in_features
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('in_channels',
                                                 container_class(in_channels))
                if module.get_mutable_attr('out_channels') is None:
                    out_channels = 0
                    if isinstance(module, nn.Conv2d):
                        out_channels = module.out_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        out_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        out_channels = module.out_features
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('out_channels',
                                                 container_class(out_channels))

    def _register_mask(self, mutable_channel: BaseMutableChannel):

        # register mutable_channel
        for channel in self.input_related + self.output_related:
            module = channel.module
            if isinstance(module, DynamicChannelMixin):
                container: MutableChannelContainer
                if channel.output_related and module.get_mutable_attr(
                        'out_channels') is not None:
                    container = module.get_mutable_attr('out_channels')
                elif channel.output_related is False \
                        and module.get_mutable_attr('in_channels') is not None:
                    container = module.get_mutable_attr('in_channels')
                else:
                    raise NotImplementedError()

                if channel.expand_ratio == 1:
                    mutable_channel_ = mutable_channel
                    start = channel.start
                    end = channel.end
                else:
                    mutable_channel_ = mutable_channel.expand_mutable_channel(
                        channel.expand_ratio)
                    start = channel.start
                    end = channel.start + (
                        channel.end - channel.start) * channel.expand_ratio
                if (start, end) in container.mutable_channels:
                    # TODO refine assert
                    existed = container.mutable_channels[(start, end)]

                    assert mutable_channel is existed \
                        or mutable_channel_ is list(
                            existed._trace_source_mutables)[0]
                else:
                    container.register_mutable(mutable_channel_, start, end)


MUTABLECHANNELGROUP = TypeVar('MUTABLECHANNELGROUP', bound=MutableChannelGroup)