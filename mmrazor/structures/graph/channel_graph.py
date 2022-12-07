# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, List

from torch.nn import Module

from mmrazor.utils import print_log
from .base_graph import BaseGraph
from .channel_flow import ChannelTensor
from .channel_nodes import (ChannelDismatchError, ChannelNode, EndNode,
                            ExpandChannelNode, InputChannelNode,
                            default_channel_node_converter)
from .module_graph import ModuleGraph, NoInputError, NoOutputError


class ChannelGraph(ModuleGraph[ChannelNode]):
    """ChannelGraph is used to trace the channel dependency of a model.

    A ChannelGraph generates a ChannelTensor as the input to the model. Then,
    the tensor can forward through all nodes and collect channel dependency.
    """

    @classmethod
    def copy_from(cls,
                  graph: 'BaseGraph',
                  node_converter: Callable = default_channel_node_converter):
        """Copy from a ModuleGraph."""
        assert isinstance(graph, ModuleGraph)
        channel_graph: ChannelGraph = super().copy_from(graph, node_converter)
        channel_graph._insert_expand_node()
        return channel_graph

    def generate_units_config(self) -> Dict:
        """Generate configs of MutableChannelUnits according to the Graph.

        "hash"{
            'init_args':{
                'num_channels': 10
            }
            'channels':{
                'input_related':[
                    {
                        "name":"backbone.bn1",
                        "start":0,
                        "end":64,
                        "expand_ratio":1,
                        "is_output_channel":false
                    }
                ],
                'output_related':[
                    ...
                ]
            }
        }
        """

        chanel_config_template: Dict = {
            'init_args': {
                'num_channels': 1
            },
            'channels': {
                'input_related': [],
                'output_related': []
            }
        }

        def process_tensor(node: ChannelNode, is_output_tensor,
                           unit_hash_dict: Dict):
            if is_output_tensor:
                tensor = node.out_channel_tensor
            else:
                tensor = node.in_channel_tensor
            assert tensor is not None
            for (start, end), hash in tensor.elems_hash_dict.items():
                channel_config = {
                    'name': node.module_name if node.is_module else node.val,
                    'start': start,
                    'end': end,
                    'is_output_channel': is_output_tensor
                }
                if hash not in unit_hash_dict:
                    unit_hash_dict[hash] = copy.deepcopy(
                        chanel_config_template)
                related_dict = unit_hash_dict[hash]['channels'][
                    'output_related' if is_output_tensor else 'input_related']
                if channel_config not in related_dict:
                    related_dict.append(channel_config)

        def fill_num_channels(units_config: Dict):

            def min_num_channels(channel_configs: List[Dict]):
                min_num_channels = int(pow(2, 32))
                for channel in channel_configs:
                    min_num_channels = min(min_num_channels,
                                           channel['end'] - channel['start'])
                return min_num_channels

            for name in units_config:
                units_config[name]['init_args'][
                    'num_channels'] = min_num_channels(
                        units_config[name]['channels']['input_related'] +
                        units_config[name]['channels']['output_related'])

        unit_hash_dict: Dict = {}
        self._reset_channel_elem_cache()
        for node in self.topo_traverse():
            process_tensor(node, True, unit_hash_dict)
            process_tensor(node, False, unit_hash_dict)
        fill_num_channels(unit_hash_dict)
        return unit_hash_dict

    def forward(self, num_input_channel=3):
        """Generate a ChanneelTensor and let it forwards through the graph."""
        for node in self.topo_traverse():
            node.reset_channel_tensors()
        for i, node in enumerate(self.topo_traverse()):
            node: ChannelNode
            if len(node.prev_nodes) == 0:
                tensor = ChannelTensor(num_input_channel)
                node.forward([tensor])
            else:
                node.forward()
        self._merge_same_module()

    # graph modification

    def _add_input_before(self, node: ChannelNode):
        """Add a input node before a ChannelNode."""
        try:
            in_channels = node.in_channels
        except Exception:
            in_channels = 3
        input_node = InputChannelNode(
            f'auto_input_{in_channels}',
            'input_placeholder',
            input_channels=in_channels)  # type: ignore
        input_node = self.add_or_find_node(input_node)
        self.connect(input_node, node)

    def _add_output_after(self, node: ChannelNode):
        """Add a output node after a ChannelNode."""

        output_node = EndNode('auto_output',
                              'output_placeholder')  # type: ignore
        output_node = self.add_or_find_node(output_node)
        self.connect(node, output_node)

    def _convert_a_node_to_end_node(self, node: ChannelNode):
        """Convert a node to end node."""

        end_node = EndNode('auto_end', 'output_placeholder')
        end_node = self.add_or_find_node(end_node)
        for prev in copy.copy(node.prev_nodes):
            self.disconnect(prev, node)
            self.connect(prev, end_node)
        self._add_input_before(node)

    def _merge_same_module(self):
        """Union all nodes with the same module to the same unit."""
        module2node: Dict[Module, List[ChannelNode]] = dict()
        for node in self:
            if isinstance(node.val,
                          Module) and len(list(node.val.parameters())) > 0:
                if node.val not in module2node:
                    module2node[node.val] = []
                if node not in module2node[node.val]:
                    module2node[node.val].append(node)

        for module in module2node:
            if len(module2node[module]) > 1:
                nodes = module2node[module]
                assert nodes[0].in_channel_tensor is not None and \
                    nodes[0].out_channel_tensor is not None
                for node in nodes[1:]:
                    nodes[0].in_channel_tensor.union(node.in_channel_tensor)
                    nodes[0].out_channel_tensor.union(node.out_channel_tensor)

    def _insert_expand_node(self):
        """Insert expand nodes in the graph."""
        num_expand_nodes = 0
        nodes: List[ChannelNode] = copy.copy(list(self.topo_traverse()))
        for node in nodes:
            try:
                node.check_channel()
            except Exception:
                for pre_node in node.prev_nodes:
                    pre_node: ChannelNode
                    if (pre_node.out_channels < node.in_channels
                            and node.in_channels % pre_node.out_channels == 0):
                        print_log(
                            (f'As the channels of {pre_node} and {node} '
                             'dismatch, we add an ExpandNode between them.'),
                            level='warning')
                        expand_ratio = (
                            node.in_channels // pre_node.out_channels)
                        # insert a expand node
                        new_node = ExpandChannelNode(
                            f'expand_{num_expand_nodes}',
                            'expand',
                            expand_ratio=expand_ratio)
                        num_expand_nodes += 1
                        self.add_node(new_node)
                        self.connect(pre_node, new_node)
                        self.connect(new_node, node)
                        self.disconnect(pre_node, node)

    # others

    def _check(self, node: ChannelNode, fix=False):
        """Helper for self.check, including check whether the Graph has any
        error and fix errors."""
        try:
            node.check_channel()
            node.check()
        except Exception as e:
            if not fix:
                raise e
            else:
                try:
                    raise e
                except NoOutputError as e:
                    print_log(f'add a output after {node}, error: {e}',
                              'debug')
                    self._add_output_after(node)
                except NoInputError as e:
                    print_log(
                        f'add a input before {node}, error: {e}',
                        level='debug')
                    self._add_input_before(node)
                except ChannelDismatchError as e:
                    print_log((f'{node} has channel error, so'
                               f'we convert it to a EndNode. error: {e}'),
                              level='debug')
                    self._convert_a_node_to_end_node(node)

                self._check(node, fix=True)

    def _reset_channel_elem_cache(self):
        """Reset hash cache of ChannelTensors."""
        # may has bug, as some tensor not recorded by node.xxxx_tensors
        for node in self.topo_traverse():
            assert (node.in_channel_tensor is not None
                    and node.out_channel_tensor is not None), f'{node}'
            node.in_channel_tensor._reset_channel_elem_cache()
            node.out_channel_tensor._reset_channel_elem_cache()
