# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines MutableChannelGroup with related modules."""
import copy
from typing import Dict, List, Tuple, Type, TypeVar, Union

import torch.nn as nn
from torch.nn import Module

from .....registry import MODELS
from .....structures.graph import ModuleGraph, ModuleNode
from .....utils.index_dict import IndexDict
from ....architectures.dynamic_op.bricks.dynamic_mixins import \
    DynamicChannelMixin
from ..simple_mutable_channel import SimpleMutableChannel

# PruneNode && PruneGraph


def is_dynamic_op(module):
    """Bool: determine if a module is a DynamicOp"""
    return isinstance(module, DynamicChannelMixin)


class PruneNode(ModuleNode):
    """Node class for pruning."""

    def __init__(self, name, obj: Module) -> None:
        super().__init__(name, obj)
        self.input_related_groups: IndexDict[ChannelGroup] = IndexDict()
        self.output_related_groups: IndexDict[ChannelGroup] = IndexDict()

    # groups operation

    def to_prune_channels(self,
                          index: Union[None, Tuple[int, int]] = None,
                          out_related=True,
                          expand_ratio=1):
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
        channels = Channel(
            self, index, out_related=out_related, expand_ratio=expand_ratio)
        return channels

    def output_channel_groups_of_prev_nodes(
            self) -> List[Dict[Tuple[int, int], 'ChannelGroup']]:
        """List[Dict[Tuple[int, int], 'ChannelGroup']: the output-related
        channel-groups of previous nodes."""
        groups = []
        for node in self.prev_nodes:
            groups.append(node.output_related_groups)
        return groups

    # channel

    @property
    def act_in_channels(self):
        """Int: activated input channel number"""
        if isinstance(self.val, nn.Module):
            if is_dynamic_op(self.val):
                self.val.mutable_in: SimpleMutableChannel
                return self.val.mutable_in.activated_channels
            else:
                if isinstance(self.val, nn.Conv2d):
                    return self.val.in_channels
                elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                    return self.val.num_features
                elif isinstance(self.val, nn.Linear):
                    return self.val.in_features
                else:
                    return None
        elif self.is_bind_node():
            assert len(self.prev_nodes) > 1, '{name} is bind node'
            return self.prev_nodes[0].act_in_channels
        elif self.is_cat_node():
            return sum([
                node.act_in_channels if node.act_in_channels is not None else 0
                for node in self.prev_nodes
            ])
        else:
            return None

    @property
    def act_out_channels(self):
        """Int: activated output channel number"""
        if isinstance(self.val, nn.Module):
            if is_dynamic_op(self.val):
                return self.val.mutable_out.activated_channels
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
            return None

    # others
    def __repr__(self) -> str:
        return f'{self.name}_{self.act_in_channels}/{self.in_channels} \
        _{self.act_out_channels}/{self.out_channels}'

    @property
    def is_prunable(self):
        """Bool: if the node prunable"""
        return self.basic_type not in ['gwconv2d']


PRUNENODE = TypeVar('PRUNENODE', bound=PruneNode)


class PruneGraph(ModuleGraph[PRUNENODE]):
    """Graph class for pruning."""

    def __init__(self) -> None:
        super().__init__()

    # groups_operation
    def colloct_groups(self) -> List['ChannelGroup']:
        """Set['ChannelGroup']: collect all channel-groups in the graph"""
        groups = []
        for node in self.topo_traverse():
            for group in node.input_related_groups.values():
                if group not in groups:
                    groups.append(group)
            for group in node.output_related_groups.values():
                if group not in groups:
                    groups.append(group)
        return groups


# ChannelGroup


class Channel:
    """PruneChannels records information about channels for pruning."""

    def __init__(self,
                 node: PruneNode,
                 index,
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
        self.node = node
        self.index = index
        self.start = index[0]
        self.end = index[1]
        self.output_related = out_related
        self.expand_ratio = expand_ratio

        self.name = node.name
        self.module: DynamicChannelMixin = node.val

    @property
    def num_channels(self):
        """Int: number of channels in the prune-channels"""
        return self.index[1] - self.index[0]

    # group related operations

    def slice(self, start, end):
        """PruneChannels: a new prune-channels who manage a slice of the channels of
        the current prune-channels"""
        channels = Channel(
            self.node,
            index=(self.start + start, self.start + end),
            out_related=self.output_related,
            expand_ratio=self.expand_ratio)
        return channels

    # others

    def __repr__(self) -> str:
        return f'{self.name}\t{self.index}\t \
        {"out" if self.output_related else "in"}\t\
        expand:{self.expand_ratio}'


class ChannelGroup:
    """A manager of prune-channels."""

    def __init__(self, num_channels) -> None:
        """
        Args:
            num_channels (int): number of prunable channels.
        """

        self.num_channels = num_channels
        self.output_related: List[Channel] = []
        self.input_related: List[Channel] = []

    # node operations

    def add_ouptut_related(self, channel: Channel):
        """None: add prune-channel which is output related"""
        assert channel.output_related
        assert self.num_channels == channel.num_channels
        if channel not in self.output_related:
            self.output_related.append(channel)

    def add_input_related(self, channel: Channel):
        """None: add prune-channel which is input related"""
        assert channel.output_related is False
        assert self.num_channels == channel.num_channels
        if channel not in self.input_related:
            self.input_related.append(channel)

    def remove_from_node(self):
        """remove information for all nodes about the group."""
        for channel in self.output_related:
            assert channel.index in channel.node.output_related_groups, \
                f'{channel.name}.{channel.index} not exist in node.out_related'
            channel.node.output_related_groups.pop(channel.index)
        for channel in self.input_related:
            assert channel.index in channel.node.input_related_groups, \
                f'{channel.name}.{channel.index} \
                    not exist in node.input_related'

            channel.node.input_related_groups.pop(channel.index)

    def apply_for_node(self):
        """register information for all nodes about the group."""
        for node in self.output_related:
            node.node.output_related_groups[node.index] = self
        for node in self.input_related:
            node.node.input_related_groups[node.index] = self

    # group operations

    @classmethod
    def union(cls, groups: List['ChannelGroup']):
        """Union channel-groups."""
        group = cls(groups[0].num_channels)
        for old_group in groups:
            for group_module in old_group.input_related:
                group.add_input_related(group_module)
            for group_module in old_group.output_related:
                group.add_ouptut_related(group_module)
        return group

    def split(self, nums: List[int]):
        """Split the channel-groups."""
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

    def slice(self, start, end):
        """Get slice of the channel-groups."""
        assert start >= 0 and end <= self.num_channels
        group = self.__class__(end - start)
        for module in self.input_related:
            group.add_input_related(module.slice(start, end))
        for module in self.output_related:
            group.add_ouptut_related(module.slice(start, end))
        return group

    # init

    @classmethod
    def parse_channel_groups(cls,
                             graph: ModuleGraph,
                             group_args={}) -> List['ChannelGroup']:
        """Parse and return all channel-groups from a module-graph."""
        group_graph = PruneGraph.copy_from(graph, PruneNode.copy_from)

        cfg = dict(type=cls.__name__, **group_args)
        groups = Graph2ChannelGroups(group_graph, cfg).groups
        for group in groups:
            group._model = graph._model
        return groups

    # to string

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

        s = f'{self.name}_\
            {len(self.output_related)},{len(self.input_related)},\
                {self.is_prunable}' + '\n'
        s += '  output_related:\n'
        s += add_prefix(list_repr(self.output_related), ' ' * 4)
        s += '  input_related\n'
        s += add_prefix(list_repr(self.input_related), ' ' * 4)
        return s


# Converter


class Graph2ChannelGroups:
    """A converter which converts a graph to a list of channel-groups."""

    def __init__(
        self,
        graph: PruneGraph,
        channel_group_cfg: Union[Dict,
                                 Type[ChannelGroup]] = ChannelGroup) -> None:
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

    def new_channel_group(self, num_channels):
        return self.channel_group_class(num_channels,
                                        **self.channel_group_args)

    def union_channel_groups(self, groups: List[ChannelGroup]):
        """List[ChannelGroup]: union a list of channel-groups"""
        group = self.channel_group_class.union(groups)
        # avoid removing multiple times
        groups_set = set(groups)
        for old_group in groups_set:
            old_group.remove_from_node()
        group.apply_for_node()
        return group

    def align_prev_output_groups(self, nodes_groups: List[Dict[Tuple[int, int],
                                                               ChannelGroup]]):
        """Align the channel-groups in the prev nodes.

            Example(pseudocode):
                >>> node1
                (0,4):group1,(4,8):group2
                >>> node2
                (0,2):group3,(2,8):group4
                >>> prev_nodes=[node1,node2]
                >>> align_prev_output_groups(prev_nodes)
                node1: (0,2):group5,(2,4):group6,(4,8):group7
                node2: (0,2):group8,(2,4):group9,(4,8):group10
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

    # operations

    def add_input_related(self,
                          group: ChannelGroup,
                          node: PruneNode,
                          index=None,
                          expand_ratio=1):
        """add some channels of a prune-node to a channel-group."""
        if index is None:
            index = (0, node.in_channels)
        group.add_input_related(
            node.to_prune_channels(
                index, out_related=False, expand_ratio=expand_ratio))
        node.input_related_groups[index] = group

    def add_output_related(self,
                           group: ChannelGroup,
                           node: PruneNode,
                           index=None,
                           expand_ratio=1):
        if index is None:
            index = (0, node.out_channels)
        group.add_ouptut_related(
            node.to_prune_channels(
                index, out_related=True, expand_ratio=expand_ratio))
        node.output_related_groups[index] = group

    # parse

    def parse_node(self, node: PruneNode):
        """parse the channels of a node, and create or update channel-
        groups."""
        pre_groups = node.output_channel_groups_of_prev_nodes()

        if node.is_mix_node():
            assert len(pre_groups) <= 1
            input_channel = node.prev_nodes[0].out_channels if len(
                node.prev_nodes) == 1 else 0
            assert input_channel == 0 or \
                node.in_channels % input_channel == 0

            current_group = self.new_channel_group(node.out_channels)
            self.add_output_related(current_group, node)

            for groups in pre_groups:
                start = 0
                for pre_group in groups.values():
                    self.add_input_related(
                        pre_group,
                        node,
                        index=(start, start + pre_group.num_channels),
                        expand_ratio=node.in_channels //
                        input_channel if input_channel != 0 else 1)
                    start += pre_group.num_channels

        elif node.is_pass_node():
            assert len(
                pre_groups
            ) <= 1, \
                (f'{node} is a pass node which should'
                 'not has more than one pre node')
            for groups in pre_groups:
                start = 0
                for group in groups.values():
                    self.add_output_related(
                        group, node, index=(start, start + group.num_channels))
                    start += group.num_channels

        elif node.is_bind_node():
            assert len(pre_groups) > 1
            for node_groups in pre_groups:
                assert len(
                    node_groups
                ) > 0, \
                    f'{node},{pre_groups} is a bind node which \
                    should have more than one pre nodes'

            # align
            self.align_prev_output_groups(pre_groups)
            pre_groups = node.output_channel_groups_of_prev_nodes()

            # union
            unoin_groups = []
            pre_groups_x = [
                list(node_groups.values()) for node_groups in pre_groups
            ]
            for j in range(len(pre_groups[0])):
                group = self.union_channel_groups(
                    [node_groups[j] for node_groups in pre_groups_x])
                unoin_groups.append(group)

            # add output_related
            start = 0
            for group in unoin_groups:
                self.add_output_related(group, node,
                                        (start, start + group.num_channels))
                start += group.num_channels

        elif node.is_cat_node():
            start = 0
            for groups in pre_groups:
                for group in groups.values():
                    self.add_output_related(
                        group, node, (start, start + group.num_channels))
                    start += group.num_channels

        else:
            raise NotImplementedError(f'{node.basic_type}')

    def parse(self, graph: PruneGraph):
        """parse a module-graph and get channel-groups."""
        for node in graph.topo_traverse():
            self.parse_node(node)
        return graph.colloct_groups()
