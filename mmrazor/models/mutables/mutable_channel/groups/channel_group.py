# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines ChannelGroup with related modules.

PruneNode                               Channel
                ------------------->
PruneGraph      Graph2ChannelGroups     ChannelGroup

PruneNode and PruneGraph are used to record the computation graph of a model.
A Channel records a slice of the input or output channels of a module.
A ChannelGroup collects all Channels with channel-dependency.
Graph2ChannelGroups is used to parse a PruneGraph and get ChannelGroups
"""

import copy
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

import torch.nn as nn
from torch.nn import Module

from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_mixins import \
    DynamicChannelMixin
from mmrazor.registry import MODELS
from mmrazor.structures.graph import ModuleGraph, ModuleNode
from mmrazor.utils import IndexDict
from ..base_mutable_channel import BaseMutableChannel

# PruneNode && PruneGraph


class PruneNode(ModuleNode):
    """Node class for pruning."""

    # init

    def __init__(self, name: str, obj: Module, module_name='') -> None:
        """
        Args:
            name (str): node name.
            obj (Module): Module
            module_name: the name of the module in the model.
        """
        super().__init__(name, obj, module_name=module_name)
        self.input_related_groups: IndexDict[ChannelGroup] = IndexDict()
        self.output_related_groups: IndexDict[ChannelGroup] = IndexDict()

    @classmethod
    def copy_from(cls, node):
        """Copy from a ModuleNode."""
        if isinstance(node, ModuleNode):
            return cls(node.name, node.val, node.module_name)
        else:
            raise NotImplementedError()

    # groups operation

    def get_channels(self,
                     index: Union[None, Tuple[int, int]] = None,
                     out_related=True,
                     expand_ratio: int = 1) -> 'Channel':
        """PruneChannels: get the channels in the node between a range

        Args:
            index (Union[None, Tuple[int, int]]): the channel range for pruning
            out_related (Bool): represents if the channels are output channels,
            otherwise input channels.
            expand_ratio (Bool): expand_ratio of the number of channels
            compared with pruning mask.
        """
        if index is None:
            index = (0, self.out_channels
                     if out_related is True else self.in_channels)
        name = self.module_name if isinstance(self.val,
                                              nn.Module) else self.name
        channel = Channel(
            name,
            self.val,
            index,
            self,
            out_related=out_related,
            expand_ratio=expand_ratio)
        return channel

    def output_related_groups_of_prev_nodes(
            self) -> List[IndexDict['ChannelGroup']]:
        """IndexDict['ChannelGroup']: the output-related
        ChannelGroups of previous nodes."""
        groups = []
        for node in self.prev_nodes:
            groups.append(node.output_related_groups)
        return groups

    # channel properties

    @property
    def act_in_channels(self) -> int:
        """Int: activated input channel number"""
        if isinstance(self.val, nn.Module):
            if isinstance(self.val, DynamicChannelMixin):
                mutable: BaseMutableChannel = self.val.get_mutable_attr(
                    'in_channels')
                return mutable.activated_channels
            else:
                if isinstance(self.val, nn.Conv2d):
                    return self.val.in_channels
                elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                    return self.val.num_features
                elif isinstance(self.val, nn.Linear):
                    return self.val.in_features
                else:
                    raise NotImplementedError()
        elif self.is_bind_node():
            assert len(self.prev_nodes) > 1, '{name} is bind node'
            return self.prev_nodes[0].act_in_channels
        elif self.is_cat_node():
            return sum([
                node.act_in_channels if node.act_in_channels is not None else 0
                for node in self.prev_nodes
            ])
        else:
            raise NotImplementedError()

    @property
    def act_out_channels(self) -> int:
        """Int: activated output channel number"""
        if isinstance(self.val, nn.Module):
            if isinstance(self.val, DynamicChannelMixin):
                mutable: BaseMutableChannel = self.val.get_mutable_attr(
                    'out_channels')
                return mutable.activated_channels
            else:
                return self.out_channels
        elif self.is_bind_node():
            assert len(self.prev_nodes) > 1, '{name} is bind node'
            return self.prev_nodes[0].act_out_channels
        elif self.is_cat_node():
            return sum([
                node.act_out_channels
                if node.act_out_channels is not None else 0
                for node in self.prev_nodes
            ])
        else:
            raise NotImplementedError()

    @property
    def is_parsed(self):
        """If this node have been parsed."""
        return len(self.input_related_groups) > 0 or len(
            self.output_related_groups) > 0

    @property
    def is_prunable(self) -> bool:
        """Bool: if the node prunable"""
        return self.basic_type not in ['gwconv2d']

    # others

    def __repr__(self) -> str:
        return (f'{self.name}_{self.act_in_channels}/{self.in_channels}'
                f'_{self.act_out_channels}/{self.out_channels}')


PRUNENODE = TypeVar('PRUNENODE', bound=PruneNode)


class PruneGraph(ModuleGraph[PRUNENODE]):
    """Graph class for pruning."""

    # init

    @classmethod
    def copy_from(cls, graph, node_converter=PruneNode.copy_from):
        """Copy from a module graph."""
        assert isinstance(graph, ModuleGraph)
        graph = super().copy_from(graph, node_converter)
        graph._merge_same_module()
        return graph

    # groups_operation

    def colloct_groups(self) -> List['ChannelGroup']:
        """List['ChannelGroup']: collect all ChannelGroups in the graph."""
        groups = []
        for node in self.topo_traverse():
            for group in node.input_related_groups.values():
                if group not in groups:
                    groups.append(group)
            for group in node.output_related_groups.values():
                if group not in groups:
                    groups.append(group)
        return groups

    # private methods

    def _merge_same_module(self):
        """Let all nodes that refer to the same module use the same
        input_related_groups and output_related_groups."""
        module2node: Dict[Any, List[PruneNode]] = dict()
        for node in self:
            if isinstance(node.val, Module):
                if node.val not in module2node:
                    module2node[node.val] = []
                if node not in module2node[node.val]:
                    module2node[node.val].append(node)
        for module in module2node:
            if len(module2node[module]) > 1:
                input_group = IndexDict()
                output_group = IndexDict()
                for node in module2node[module]:
                    node.input_related_groups = input_group
                    node.output_related_groups = output_group


# Channel && ChannelGroup


class Channel:
    """Channel records information about channels for pruning."""

    # init

    def __init__(self,
                 name,
                 module,
                 index,
                 node: PruneNode = None,
                 out_related=True,
                 expand_ratio=1) -> None:
        """
        Args:
            node: (PruneNode): prune-node to be recorded
            index (Union[None, Tuple[int, int]]): the channel range for pruning
            out_related (Bool): represents if the channels are output channels,
            otherwise input channels
            expand_ratio (Bool): expand_ratio of the number of channels
            compared with pruning mask
        """
        self.name = name
        self.module: DynamicChannelMixin = module
        self.index = index
        self.start = index[0]
        self.end = index[1]

        self.node = node

        self.output_related = out_related
        self.expand_ratio = expand_ratio

    @classmethod
    def init_using_cfg(cls, model: nn.Module, config: Dict):
        """init a Channel using a config which can be generated by
        self.config_template()"""
        name = config['name']
        start = config['start']
        end = config['end']
        expand_ratio = config['expand_ratio']
        is_output = config['is_output_related']

        name2module = dict(model.named_modules())
        name2module.pop('')
        module = name2module[name] if name in name2module else None
        return Channel(
            name,
            module, (start, end),
            out_related=is_output,
            expand_ratio=expand_ratio)

    # config template
    def config_template(self):
        """Generate a config template which can be used to initialize a Channel
        by cls.init_using_cfg(**kwargs)"""
        return {
            'name': self.name,
            'start': self.start,
            'end': self.end,
            'expand_ratio': self.expand_ratio,
            'is_output_related': self.output_related
        }

    # basic properties

    @property
    def num_channels(self) -> int:
        """Int: number of channels in the Channels"""
        return self.index[1] - self.index[0]

    @property
    def is_prunable(self) -> bool:
        """If the channel is prunable."""
        if isinstance(self.module, nn.Conv2d):
            # group-wise conv
            if self.module.groups != 1 and not (self.module.groups ==
                                                self.module.in_channels ==
                                                self.module.out_channels):
                return False
        return True

    # node operations

    def slice(self, start: int, end: int) -> 'Channel':
        """Channel: a new Channel who manage a slice of the current Channel."""
        channel = Channel(
            name=self.name,
            module=self.module,
            index=(self.start + start, self.start + end),
            node=self.node,
            out_related=self.output_related,
            expand_ratio=self.expand_ratio)
        return channel

    # others

    def __repr__(self) -> str:
        return f'{self.name}\t{self.index}\t \
        {"out" if self.output_related else "in"}\t\
        expand:{self.expand_ratio}'


@MODELS.register_module()
class ChannelGroup:
    """A manager for Channels."""

    def __init__(self, num_channels: int) -> None:
        """
        Args:
            num_channels (int): the dimension of Channels.
        """

        self.num_channels = num_channels
        self.output_related: List[Channel] = []
        self.input_related: List[Channel] = []
        self.init_args: Dict = {
        }  # is used to generate new channel group with same args

    @classmethod
    def init_using_cfg(cls, model: nn.Module, config: Dict):
        """init a ChannelGroup using a config which can be generated by
        self.config_template()"""
        config = copy.deepcopy(config)
        if 'channels' in config:
            channels = config.pop('channels')
        else:
            channels = None
        group = cls(**(config['init_args']))
        if channels is not None:
            for channel_config in channels['input_related']:
                group.add_input_related(
                    Channel.init_using_cfg(model, channel_config))
            for channel_config in channels['output_related']:
                group.add_ouptut_related(
                    Channel.init_using_cfg(model, channel_config))
        return group

    @classmethod
    def parse_channel_groups(cls,
                             graph: ModuleGraph,
                             group_args={}) -> List['ChannelGroup']:
        """Parse a module-graph and get ChannelGroups."""
        group_graph = PruneGraph.copy_from(graph, PruneNode.copy_from)

        cfg = dict(type=cls.__name__, **group_args)
        groups = Graph2ChannelGroups(group_graph, cfg).groups
        for group in groups:
            group._model = graph._model
        return groups

    # basic property

    @property
    def name(self) -> str:
        """str: name of the group"""
        first_module = self.output_related[0] if len(
            self.output_related) > 0 else self.input_related[0]
        name = f'{first_module.name}_{first_module.index}_'
        name += f'out_{len(self.output_related)}_in_{len(self.input_related)}'
        return name

    # config template

    def config_template(self, with_init_args=False, with_channels=False):
        """Generate a config template which can be used to initialize a
        ChannelGroup by cls.init_using_cfg(**kwargs)"""
        config = {}
        if with_init_args:
            config['init_args'] = {'num_channels': self.num_channels}
        if with_channels:
            config['channels'] = self._channel_dict()
        return config

    # node operations

    def add_ouptut_related(self, channel: Channel):
        """None: add a Channel which is output related"""
        assert channel.output_related
        assert self.num_channels == channel.num_channels
        if channel not in self.output_related:
            self.output_related.append(channel)

    def add_input_related(self, channel: Channel):
        """None: add a Channel which is input related"""
        assert channel.output_related is False
        assert self.num_channels == channel.num_channels
        if channel not in self.input_related:
            self.input_related.append(channel)

    def remove_from_node(self):
        """Remove recorded information in all nodes about this group."""
        for channel in self.output_related:
            assert channel.node is not None \
                and channel.index in channel.node.output_related_groups, \
                f'{channel.name}.{channel.index} not exist in node.out_related'
            channel.node.output_related_groups.pop(channel.index)
        for channel in self.input_related:
            assert channel.node is not None \
                and channel.index in channel.node.input_related_groups, \
                f'{channel.name}.{channel.index} \
                    not exist in node.input_related'

            channel.node.input_related_groups.pop(channel.index)

    def apply_for_node(self):
        """Register the information about this group for all nodes."""
        for channel in self.output_related:
            assert channel.node is not None
            channel.node.output_related_groups[channel.index] = self
        for channel in self.input_related:
            assert channel.node is not None
            channel.node.input_related_groups[channel.index] = self

    # group operations

    @classmethod
    def union(cls, groups: List['ChannelGroup']) -> 'ChannelGroup':
        """ChannelGroup: Union ChannelGroups and return."""
        group = cls(groups[0].num_channels,
                    **groups[0].init_args)  # type: ignore
        for old_group in groups:
            for group_module in old_group.input_related:
                group.add_input_related(group_module)
            for group_module in old_group.output_related:
                group.add_ouptut_related(group_module)
        return group

    def split(self, nums: List[int]) -> List['ChannelGroup']:
        """Split the ChannelGroup and return."""
        assert sum(nums) == self.num_channels

        if len(nums) == 1:
            return [self]
        else:
            groups = []
            start = 0
            for num in nums:
                groups.append(self.slice(start, start + num))
                start += num
            return groups

    def slice(self, start: int, end: int) -> 'ChannelGroup':
        """Get a slice of the ChannelGroup."""
        assert start >= 0 and end <= self.num_channels
        group = self.__class__(end - start, **self.init_args)  # type: ignore
        for module in self.input_related:
            group.add_input_related(module.slice(start, end))
        for module in self.output_related:
            group.add_ouptut_related(module.slice(start, end))
        return group

    # others

    def __repr__(self):

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
             f'\t{len(self.output_related)},{len(self.input_related)}'
             f'\t{self.is_prunable}\n')
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


# Group to ChannelGroup Converter


class Graph2ChannelGroups:
    """A converter which converts a Graph to a list of ChannelGroups."""

    def __init__(
        self,
        graph: PruneGraph,
        channel_group_cfg: Union[Dict,
                                 Type[ChannelGroup]] = ChannelGroup) -> None:
        """
        Args:
            graph (PruneGraph): input prune-graph
            channel_group_cfg: the config for generating groups
        """
        self.graph = graph
        if isinstance(channel_group_cfg, dict):
            self.channel_group_class = MODELS.module_dict[
                channel_group_cfg['type']]
            self.channel_group_args = copy.copy(channel_group_cfg)
            self.channel_group_args.pop('type')
        else:
            self.channel_group_class = channel_group_cfg
            self.channel_group_args = {}
        self.groups = self.parse(self.graph)

    # group operations

    def new_channel_group(self, num_channels) -> ChannelGroup:
        """Initialize a ChannelGroup."""
        return self.channel_group_class(num_channels,
                                        **self.channel_group_args)

    def union_node_groups(
            self,
            node_groups_list=List[IndexDict[ChannelGroup]]
    ) -> List[ChannelGroup]:
        """Union groups of nodes."""
        union_groups = []
        for index in copy.copy(node_groups_list[0]):
            groups = [node_groups[index] for node_groups in node_groups_list]
            group = self.union_groups(groups)
            union_groups.append(group)
        return union_groups

    def union_groups(self, groups: List[ChannelGroup]) -> ChannelGroup:
        """List[ChannelGroup]: union a list of ChannelGroups"""
        group = self.channel_group_class.union(groups)
        # avoid removing multiple times
        groups_set = set(groups)
        for old_group in groups_set:
            old_group.remove_from_node()
        group.apply_for_node()
        return group

    def align_node_groups(self, nodes_groups: List[IndexDict[ChannelGroup]]):
        """Align the ChannelGroups in the prev nodes.

            Example(pseudocode):
                >>> node1
                (0,4):group1, (4,8):group2
                >>> node2
                (0,2):group3, (2,8):group4
                >>> prev_nodes=[node1,node2]
                >>> align_prev_output_groups(prev_nodes)
                node1: (0,2):group5, (2,4):group6, (4,8):group7
                node2: (0,2):group8, (2,4):group9, (4,8):group10
        """

        def points2nums(points):
            nums = [points[i + 1] - points[i] for i in range(len(points) - 1)]
            return nums

        # get points
        points = set()
        for node_groups in nodes_groups:
            start = 0
            for group in node_groups.values():
                points.add(start)
                points.add(start + group.num_channels)
                start += group.num_channels
        points_list = list(points)
        points_list.sort()

        # split group
        new_groups: List[ChannelGroup] = []
        old_groups: List[ChannelGroup] = []
        for node_groups in nodes_groups:
            start = 0
            for group in node_groups.values():
                end = start + group.num_channels
                in_points = [
                    point for point in points_list if start <= point <= end
                ]
                in_nums = points2nums(in_points)
                if len(in_nums) == 1:
                    pass
                else:
                    split_groups = group.split(in_nums)
                    new_groups.extend(split_groups)
                    old_groups.append(group)
                start = end

        # apply
        for group in old_groups:
            group.remove_from_node()
        for group in new_groups:
            group.apply_for_node()

    # node operations

    def add_input_related(self,
                          group: ChannelGroup,
                          node: PruneNode,
                          index: Tuple[int, int] = None,
                          expand_ratio: int = 1):
        """Add a Channel of a PruneNode to a the input-related channels of a
        ChannelGroup."""
        if index is None:
            index = (0, node.in_channels)
        group.add_input_related(
            node.get_channels(
                index, out_related=False, expand_ratio=expand_ratio))
        node.input_related_groups[index] = group

    def add_output_related(self,
                           group: ChannelGroup,
                           node: PruneNode,
                           index: Tuple[int, int] = None,
                           expand_ratio=1):
        """Add a Channel of a PruneNode to a the output-related channels of a
        ChannelGroup."""
        if index is None:
            index = (0, node.out_channels)
        group.add_ouptut_related(
            node.get_channels(
                index, out_related=True, expand_ratio=expand_ratio))
        node.output_related_groups[index] = group

    # parse

    def parse_node(self, node: PruneNode):
        """Parse the channels of a node, and create or update ChannelGroups."""
        prev_node_groups = node.output_related_groups_of_prev_nodes()

        if node.is_parsed:

            # align
            self.align_node_groups(prev_node_groups +
                                   [node.input_related_groups])

            # union
            prev_node_groups = node.output_related_groups_of_prev_nodes()
            self.union_node_groups(prev_node_groups +
                                   [node.input_related_groups])

        elif node.is_mix_node():
            assert len(prev_node_groups) <= 1
            input_channel = node.prev_nodes[0].out_channels if len(
                node.prev_nodes) == 1 else 0
            assert input_channel == 0 or \
                node.in_channels % input_channel == 0

            # new group and add output-related
            current_group = self.new_channel_group(node.out_channels)
            self.add_output_related(current_group, node)

            # add input-related
            for node_groups in prev_node_groups:
                start = 0
                for group in node_groups.values():
                    self.add_input_related(
                        group,
                        node,
                        index=(start, start + group.num_channels),
                        expand_ratio=node.in_channels //
                        input_channel if input_channel != 0 else 1)
                    start += group.num_channels

        elif node.is_pass_node():
            assert len(prev_node_groups) <= 1, \
                (f'{node} is a pass node which should'
                 'not has more than one pre node')

            # add input-related and output-related
            for node_groups in prev_node_groups:
                start = 0
                for group in node_groups.values():
                    self.add_output_related(
                        group, node, index=(start, start + group.num_channels))
                    self.add_input_related(
                        group, node, index=(start, start + group.num_channels))
                    start += group.num_channels

        elif node.is_bind_node():
            assert len(prev_node_groups) > 1
            for node_groups in prev_node_groups:
                assert len(node_groups) > 0, \
                    f'{node},{prev_node_groups} is a bind node which \
                    should have more than one pre nodes'

            # align
            self.align_node_groups(prev_node_groups)

            # union
            unoin_groups = self.union_node_groups(prev_node_groups)

            # add output-related
            start = 0
            for group in unoin_groups:
                self.add_output_related(group, node,
                                        (start, start + group.num_channels))
                start += group.num_channels

        elif node.is_cat_node():
            # add output-related
            start = 0
            for node_groups in prev_node_groups:
                for group in node_groups.values():
                    self.add_output_related(
                        group, node, (start, start + group.num_channels))
                    start += group.num_channels

        else:
            raise NotImplementedError(f'{node.basic_type}')

    def parse(self, graph: PruneGraph):
        """Parse a module-graph and get ChannelGroups."""
        for node in graph.topo_traverse():
            self.parse_node(node)
        return graph.colloct_groups()
