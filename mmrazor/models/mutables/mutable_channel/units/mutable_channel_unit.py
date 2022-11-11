# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines MutableChannelUnit."""
import abc
from collections import Set
from typing import Dict, List, Type, TypeVar

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables import DerivedMutable
from mmrazor.models.mutables.mutable_channel import (BaseMutableChannel,
                                                     MutableChannelContainer)
from mmrazor.models.mutables.mutable_value import MutableValue
from .channel_unit import Channel, ChannelUnit


class MutableChannelUnit(ChannelUnit):
    # init methods
    def __init__(self, num_channels: int, **kwargs) -> None:
        """MutableChannelUnit inherits from ChannelUnit, which manages channels
        with channel-dependency. Compared with ChannelUnit, MutableChannelUnit
        defines the core interfaces for pruning. By inheriting
        MutableChannelUnit, we can implement a variant pruning and nas
        algorithm. These apis includes.

            - basic property
                - name
                - is_mutable
            - before pruning
                - prepare_for_pruning
            - pruning stage
                - current_choice
                - sample_choice
            - after pruning
                - fix_chosen

        Args:
            num_channels (int): dimension of the channels of the Channel
            objects in the unit.
        """

        super().__init__(num_channels)

    @classmethod
    def init_from_mutable_channel(cls, mutable_channel: BaseMutableChannel):
        unit = cls(mutable_channel.num_channels)
        return unit

    @classmethod
    def init_from_predefined_model(cls, model: nn.Module):
        """Initialize units using the model with pre-defined dynamicops and
        mutable-channels."""

        def process_container(contanier: MutableChannelContainer,
                              module,
                              module_name,
                              mutable2units,
                              is_output=True):
            for index, mutable in contanier.mutable_channels.items():
                expand_ratio = 1
                if isinstance(mutable, DerivedMutable):
                    source_mutables: Set = \
                        mutable._trace_source_mutables()
                    source_channel_mutables = [
                        mutable for mutable in source_mutables
                        if isinstance(mutable, BaseMutableChannel)
                    ]
                    assert len(source_channel_mutables) == 1, (
                        'only support one mutable channel '
                        'used in DerivedMutable')
                    mutable = list(source_channel_mutables)[0]

                    source_value_mutables = [
                        mutable for mutable in source_mutables
                        if isinstance(mutable, MutableValue)
                    ]
                    assert len(source_value_mutables) <= 1, (
                        'only support one mutable value '
                        'used in DerivedMutable')
                    expand_ratio = int(
                        list(source_value_mutables)
                        [0].current_choice) if source_value_mutables else 1

                if mutable not in mutable2units:
                    mutable2units[mutable] = cls.init_from_mutable_channel(
                        mutable)

                unit: MutableChannelUnit = mutable2units[mutable]
                if is_output:
                    unit.add_ouptut_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output,
                            expand_ratio=expand_ratio))
                else:
                    unit.add_input_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output,
                            expand_ratio=expand_ratio))

        mutable2units: Dict = {}
        for name, module in model.named_modules():
            if isinstance(module, DynamicChannelMixin):
                in_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'in_channels')
                out_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'out_channels')
                process_container(in_container, module, name, mutable2units,
                                  False)
                process_container(out_container, module, name, mutable2units,
                                  True)
        units = list(mutable2units.values())
        return units

    # properties

    @property
    def is_mutable(self) -> bool:
        """If the channel-unit is prunable."""

        def traverse(channels: List[Channel]):
            has_dynamic_op = False
            all_channel_prunable = True
            for channel in channels:
                if channel.is_mutable is False:
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

    def config_template(self,
                        with_init_args=False,
                        with_channels=False) -> Dict:
        """Return the config template of this unit. By default, the config
        template only includes a key 'choice'.

        Args:
            with_init_args (bool): if the config includes args for
                initialization.
            with_channels (bool): if the config includes info about
                channels. the config with info about channels can used to
                parse channel units without tracer.
        """
        config = super().config_template(with_init_args, with_channels)
        config['choice'] = self.current_choice
        return config

    # before pruning: prepare a model

    @abc.abstractmethod
    def prepare_for_pruning(self, model):
        """Post process after parse units.

        For example, we need to register mutables to dynamic-ops.
        """
        raise NotImplementedError

    # pruning: choice-related

    @property
    def current_choice(self):
        """Choice of this unit."""
        raise NotImplementedError()

    @current_choice.setter
    def current_choice(self, choice) -> None:
        """setter of current_choice."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_choice(self):
        """Randomly sample a valid choice and return."""
        raise NotImplementedError()

    # after pruning

    def fix_chosen(self, choice=None):
        """Make the channels in this unit fixed."""
        if choice is not None:
            self.current_choice = choice

    # private methods

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

        for channel in list(self.input_related) + list(self.output_related):
            if isinstance(channel.module, nn.Module):
                module = get_module(model, channel.name)
                if type(module) in dynamicop_map:
                    new_module = dynamicop_map[type(module)].convert_from(
                        module)
                    replace_op(model, channel.name, new_module)
                    channel.module = new_module
                else:
                    channel.module = module

    @staticmethod
    def _register_channel_container(
            model: nn.Module, container_class: Type[MutableChannelContainer]):
        """register channel container for dynamic ops."""
        for module in model.modules():
            if isinstance(module, DynamicChannelMixin):
                in_channels = getattr(module,
                                      module.attr_mappings['in_channels'], 0)
                if module.get_mutable_attr('in_channels') is None:
                    module.register_mutable_attr('in_channels',
                                                 container_class(in_channels))
                out_channels = getattr(module,
                                       module.attr_mappings['out_channels'], 0)
                if module.get_mutable_attr('out_channels') is None:

                    module.register_mutable_attr('out_channels',
                                                 container_class(out_channels))

    def _register_mutable_channel(self, mutable_channel: BaseMutableChannel):
        # register mutable_channel
        for channel in list(self.input_related) + list(self.output_related):
            module = channel.module
            if isinstance(module, DynamicChannelMixin):
                container: MutableChannelContainer
                if channel.is_output_channel and module.get_mutable_attr(
                        'out_channels') is not None:
                    container = module.get_mutable_attr('out_channels')
                elif channel.is_output_channel is False \
                        and module.get_mutable_attr('in_channels') is not None:
                    container = module.get_mutable_attr('in_channels')
                else:
                    raise NotImplementedError()

                if channel.num_channels == self.num_channels:
                    mutable_channel_ = mutable_channel
                    start = channel.start
                    end = channel.end
                elif channel.num_channels > self.num_channels:
                    if channel.num_channels % self.num_channels == 0:
                        mutable_channel_ = \
                            mutable_channel.expand_mutable_channel(
                                channel.num_channels // self.num_channels)
                        start = channel.start
                        end = channel.end
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

                if (start, end) in container.mutable_channels:
                    existed = container.mutable_channels[(start, end)]
                    if not isinstance(existed, DerivedMutable):
                        assert mutable_channel is existed
                    else:
                        source_mutables = list(
                            existed._trace_source_mutables())
                        is_same = [
                            mutable_channel is mutable
                            for mutable in source_mutables
                        ]
                        assert any(is_same)

                else:
                    container.register_mutable(mutable_channel_, start, end)


ChannelUnitType = TypeVar('ChannelUnitType', bound=MutableChannelUnit)
