# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List

from torch.nn import Module

from .base_graph import BaseGraph
from .channel_modules import BaseChannelUnit, ChannelTensor
from .channel_nodes import ChannelNode, default_channel_node_converter
from .module_graph import ModuleGraph


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
        return super().copy_from(graph, node_converter)

    def collect_units(self) -> List[BaseChannelUnit]:
        """Collect channel units in the graph."""
        units = list()
        for node in self.topo_traverse():
            node.register_channel_to_units()
        for node in self.topo_traverse():
            for unit in node.in_channel_tensor.unit_list + \
                    node.out_channel_tensor.unit_list:
                if unit not in units:
                    units.append(unit)
        return units

    def forward(self, num_input_channel=3):
        """Generate a ChanneelTensor and let it forwards through the graph."""
        for node in self.topo_traverse():
            node.reset_channel_tensors()
        self._merge_same_module()
        for i, node in enumerate(self.topo_traverse()):
            node: ChannelNode
            if len(node.prev_nodes) == 0:
                channel_list = ChannelTensor(num_input_channel)
                node.forward([channel_list])
            else:
                node.forward()

    def _merge_same_module(self):
        """Union all nodes with the same module to the same unit."""
        module2node: Dict[Module, List[ChannelNode]] = dict()
        for node in self:
            if isinstance(node.val, Module):
                if node.val not in module2node:
                    module2node[node.val] = []
                if node not in module2node[node.val]:
                    module2node[node.val].append(node)

        for module in module2node:
            if len(module2node[module]) > 1:
                nodes = module2node[module]
                input_channel_tensor = ChannelTensor(nodes[0].in_channels)
                out_channel_tensor = ChannelTensor(nodes[0].out_channels)
                for node in nodes:
                    ChannelTensor.union(input_channel_tensor,
                                        node.in_channel_tensor)
                    ChannelTensor.union(out_channel_tensor,
                                        node.out_channel_tensor)
