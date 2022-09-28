# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List

import torch.nn as nn
from mmengine.model import BaseModule

from mmrazor.structures.graph import ModuleGraph
from mmrazor.structures.graph.channel_graph import ChannelGraph
from mmrazor.structures.graph.channel_modules import (BaseChannel,
                                                      BaseChannelGroup)
from mmrazor.structures.graph.channel_nodes import \
    default_channel_node_converter


class Channel(BaseChannel):
    """Channel records information about channels for pruning.

    Args:
        name (str): The name of the channel. When the channel is related with
            a module, the name should be the name of the module in the model.
        module (Any): Module of the channel.
        index (Tuple[int,int]): Index(start,end) of the Channel in the Module
        node (ChannelNode, optional): A ChannelNode corresponding to the
            Channel. Defaults to None.
        is_output_channel (bool, optional): Is the channel output channel.
            Defaults to True.
        expand_ratio (int, optional): Expand ratio of the mask. Defaults to 1.
    """

    # init

    def __init__(self,
                 name,
                 module,
                 index,
                 node=None,
                 is_output_channel=True,
                 expand_ratio=1) -> None:
        super().__init__(name, module, index, node, is_output_channel,
                         expand_ratio)

    @classmethod
    def init_from_cfg(cls, model: nn.Module, config: Dict):
        """init a Channel using a config which can be generated by
        self.config_template()"""
        name = config['name']
        start = config['start']
        end = config['end']
        expand_ratio = config['expand_ratio'] \
            if 'expand_ratio' in config else 1
        is_output_channel = config['is_output_channel']

        name2module = dict(model.named_modules())
        name2module.pop('')
        module = name2module[name] if name in name2module else None
        return Channel(
            name,
            module, (start, end),
            is_output_channel=is_output_channel,
            expand_ratio=expand_ratio)

    @classmethod
    def init_from_base_channel(cls, base_channel: BaseChannel):
        """Init from a BaseChannel object."""
        return cls(
            base_channel.name,
            base_channel.module,
            base_channel.index,
            node=None,
            is_output_channel=base_channel.is_output_channel,
            expand_ratio=base_channel.expand_ratio)

    # config template

    def config_template(self):
        """Generate a config template which can be used to initialize a Channel
        by cls.init_from_cfg(**kwargs)"""
        return {
            'name': self.name,
            'start': self.start,
            'end': self.end,
            'expand_ratio': self.expand_ratio,
            'is_output_channel': self.is_output_channel
        }

    # basic properties

    @property
    def is_mutable(self) -> bool:
        """If the channel is prunable."""
        if isinstance(self.module, nn.Conv2d):
            # group-wise conv
            if self.module.groups != 1 and not (self.module.groups ==
                                                self.module.in_channels ==
                                                self.module.out_channels):
                return False
        return True


# Channel && ChannelGroup


class ChannelGroup(BaseModule):
    """A group of Channels.

    A ChannelGroup has two list, input_related and output_related, to store
    the Channels. These Channels are dependent on each other, and have to
    have the same number of activated number of channels.

    Args:
        num_channels (int): the number of channels of Channel object.
    """

    # init methods

    def __init__(self, num_channels: int, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.output_related: List[Channel] = []
        self.input_related: List[Channel] = []
        self.init_args: Dict = {
        }  # is used to generate new channel group with same args

    @classmethod
    def init_from_cfg(cls, model: nn.Module, config: Dict) -> 'ChannelGroup':
        """init a ChannelGroup using a config which can be generated by
        self.config_template()"""

        def auto_fill_channel_config(channel_config: Dict,
                                     is_output_channel: bool,
                                     group_config: Dict = config):
            if 'start' not in channel_config:
                channel_config['start'] = 0
            if 'end' not in channel_config:
                channel_config['end'] = group_config['init_args'][
                    'num_channels']
            channel_config['is_output_channel'] = is_output_channel

        config = copy.deepcopy(config)
        if 'channels' in config:
            channels = config.pop('channels')
        else:
            channels = None
        group = cls(**(config['init_args']))
        if channels is not None:
            for channel_config in channels['input_related']:
                auto_fill_channel_config(channel_config, False)
                group.add_input_related(
                    Channel.init_from_cfg(model, channel_config))
            for channel_config in channels['output_related']:
                auto_fill_channel_config(channel_config, True)
                group.add_ouptut_related(
                    Channel.init_from_cfg(model, channel_config))
        return group

    @classmethod
    def init_from_channel_group(cls,
                                group: 'ChannelGroup',
                                args: Dict = {}) -> 'ChannelGroup':
        """Initial a object of current class from a ChannelGroup object.

        Args:
            group (ChannelGroup)
            args (Dict): arguments to initial the object of current class.

        Returns:
            Type(cls)
        """
        """Initialize a MutalbeChannelGroup from a ChannelGroup."""
        args['num_channels'] = group.num_channels
        mutable_group = cls(**args)
        mutable_group.input_related = group.input_related
        mutable_group.output_related = group.output_related
        return mutable_group

    @classmethod
    def init_from_graph(cls,
                        graph: ModuleGraph,
                        group_args={}) -> List['ChannelGroup']:
        """Parse a module-graph and get ChannelGroups."""

        def init_from_base_channel_group(base_channel_group: BaseChannelGroup):
            group = cls(len(base_channel_group.channel_elems), **group_args)
            group.input_related = [
                Channel.init_from_base_channel(channel)
                for channel in base_channel_group.input_related
            ]
            group.output_related = [
                Channel.init_from_base_channel(channel)
                for channel in base_channel_group.output_related
            ]
            return group

        group_graph = ChannelGraph.copy_from(graph,
                                             default_channel_node_converter)
        group_graph.forward()
        groups = group_graph.collect_groups()
        groups = [init_from_base_channel_group(group) for group in groups]
        return groups

    # tools

    @property
    def name(self) -> str:
        """str: name of the group"""
        if len(self.output_related) + len(self.input_related) > 0:
            first_module = (self.output_related + self.input_related)[0]
            first_module_name = f'{first_module.name}_{first_module.index}'
        else:
            first_module_name = 'groupx'
        name = f'{first_module_name}_{self.num_channels}'
        return name

    def config_template(self,
                        with_init_args=False,
                        with_channels=False) -> Dict:
        """Generate a config template which can be used to initialize a
        ChannelGroup by cls.init_from_cfg(**kwargs)"""
        config = {}
        if with_init_args:
            config['init_args'] = {'num_channels': self.num_channels}
        if with_channels:
            config['channels'] = self._channel_dict()
        return config

    # node operations

    def add_ouptut_related(self, channel: Channel):
        """Add a Channel which is output related."""
        assert channel.is_output_channel
        assert self.num_channels == channel.num_channels
        if channel not in self.output_related:
            self.output_related.append(channel)

    def add_input_related(self, channel: Channel):
        """Add a Channel which is input related."""
        assert channel.is_output_channel is False
        assert self.num_channels == channel.num_channels
        if channel not in self.input_related:
            self.input_related.append(channel)

    # others

    def __repr__(self) -> str:

        def add_prefix(string: str, prefix='  '):
            str_list = string.split('\n')
            str_list = [
                prefix + line if line != '' else line for line in str_list
            ]
            return '\n'.join(str_list)

        def list_repr(lit: List):
            s = '[\n'
            for item in lit:
                s += add_prefix(item.__repr__(), '  ') + '\n'
            s += ']\n'
            return s

        s = (f'{self.name}_'
             f'\t{len(self.output_related)},{len(self.input_related)}\n')
        s += '  output_related:\n'
        s += add_prefix(list_repr(self.output_related), ' ' * 4)
        s += '  input_related\n'
        s += add_prefix(list_repr(self.input_related), ' ' * 4)
        return s

    # private methods

    def _channel_dict(self) -> Dict:
        """Return channel config."""
        info = {
            'input_related':
            [channel.config_template() for channel in self.input_related],
            'output_related':
            [channel.config_template() for channel in self.output_related],
        }
        return info
