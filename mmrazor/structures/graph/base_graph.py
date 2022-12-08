# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines BaseNode and BaseGraph, which are used to model Directed
Acyclic Graph(DAG)"""
from collections import OrderedDict
from copy import copy
from typing import Any, Callable, Generic, Iterator, List, TypeVar

# BaseNode && BaseGraph


class BaseNode:
    """A single node in a graph."""

    def __init__(self, name: str, val: Any) -> None:
        """
        Args:
            name (str): name of the node.
            val (any): content of the node.
        """
        self.val = val
        self.name = name
        self.prev_nodes: List = []
        self.next_nodes: List = []

    # node operation

    def add_prev_node(self, node: 'BaseNode'):
        """add previous node."""
        if node not in self.prev_nodes:
            self.prev_nodes.append(node)
        if self not in node.next_nodes:
            node.next_nodes.append(self)

    def add_next_node(self, node: 'BaseNode'):
        """add next node."""
        if node not in self.next_nodes:
            self.next_nodes.append(node)
        if self not in node.prev_nodes:
            node.prev_nodes.append(self)

    @classmethod
    def copy_from(cls, node: 'BaseNode'):
        """Copy a node, and generate a new node with current node type."""
        return cls(node.name, node.val)

    # compare operation

    def __hash__(self) -> int:
        """Hash the node."""
        return hash((self.val, self.name))

    def __eq__(self, other):
        """Compare two nodes."""
        return self.val is other.val and self.name == other.name

    # other

    def __repr__(self) -> str:
        return self.name


BASENODE = TypeVar('BASENODE', bound=BaseNode)


class BaseGraph(Generic[BASENODE]):
    """A Directed Acyclic Graph(DAG)"""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: OrderedDict[str, BASENODE] = OrderedDict()

    # graph operations

    @classmethod
    def copy_from(cls,
                  graph: 'BaseGraph',
                  node_converter: Callable = BaseNode.copy_from):
        """Copy a graph, and generate a new graph of the current class.

        Args:
            graph (BaseGraph): the graph to be copied.
            node_converter (Callable): a function that converts node,
            when coping graph.
        """
        old2new = {}
        new_graph = cls()
        # copy nodes
        for old in graph:
            old2new[old] = new_graph.add_or_find_node(node_converter(old))

        # connect
        for old in graph:
            for pre in old.prev_nodes:
                new_graph.connect(old2new[pre], old2new[old])
        return new_graph

    # node operations

    def add_or_find_node(self, node: BASENODE):
        """Add a node to the graph.

        If the node has exsited in the graph, the function will return the node
        recorded in the graph.
        """
        find = self.find_node(node)
        if find is not None:
            return find
        else:
            self.add_node(node)
            return node

    def find_node(self, node: BaseNode):
        """Find a node and return."""
        if node.name in self.nodes and node.val == self.nodes[node.name].val:
            return self.nodes[node.name]
        else:
            return None

    def add_node(self, node: BASENODE):
        """Add a node."""
        if node.name not in self.nodes:
            self.nodes[node.name] = node
        else:
            raise Exception(f'{node.name} already exists in graph')

    def connect(self, pre_node: BASENODE, next_node: BASENODE):
        """Add an edge from pre_node to next_node."""
        pre_node_ = self.find_node(pre_node)
        next_node_ = self.find_node(next_node)
        assert pre_node_ is not None and next_node_ is not None, \
            f"{pre_node},{next_node} don't exist in the graph."
        pre_node = pre_node_
        next_node = next_node_
        pre_node.add_next_node(next_node)
        next_node.add_prev_node(pre_node)

    def disconnect(self, pre_node: BASENODE, next_node: BASENODE):
        """Remove the edge form pre_node to next_node."""
        pre_node_ = self.find_node(pre_node)
        next_node_ = self.find_node(next_node)
        assert pre_node_ is not None and next_node_ is not None, \
            f"{pre_node},{next_node} don't exist in the graph."
        pre_node = pre_node_
        next_node = next_node_
        if next_node in pre_node.next_nodes:
            pre_node.next_nodes.remove(next_node)
        if pre_node in next_node.prev_nodes:
            next_node.prev_nodes.remove(pre_node)

    def delete_node(self, node: BASENODE):
        """Delete a node with its related edges."""
        node = self.find_node(node)
        assert node is not None

        if len(node.prev_nodes) == 0:
            for next in copy(node.next_nodes):
                self.disconnect(node, next)
        elif len(node.next_nodes) == 0:
            for pre in copy(node.prev_nodes):
                self.disconnect(pre, node)
        elif len(node.prev_nodes) == 1:
            pre_node = node.prev_nodes[0]
            self.disconnect(pre_node, node)
            for next in copy(node.next_nodes):
                self.disconnect(node, next)
                self.connect(pre_node, next)
        elif len(node.next_nodes) == 1:
            next_node = node.next_nodes[0]
            self.disconnect(node, next_node)
            for pre in copy(node.prev_nodes):
                self.disconnect(pre, node)
                self.connect(pre, next_node)
        else:
            raise Exception(f'not delete {node}, \
                as it has more than one inputs and outputs')
        self.nodes.pop(node.name)

    # work as a collection

    def __iter__(self) -> Iterator[BASENODE]:
        """Traverse all nodes in the graph."""
        for x in self.nodes.values():
            yield x

    def __contains__(self, node: BASENODE) -> bool:
        """Check if a node is contained in the graph."""
        return node.name in self.nodes

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    # other

    def __repr__(self):
        res = f'Graph with {len(self)} nodes:\n'
        for node in self:
            res += '{0:<80} -> {1:^80} -> {2:<80}\n'.format(
                str(node.prev_nodes), node.__repr__(), str(node.next_nodes))
        return res

    # traverse

    def topo_traverse(self) -> Iterator[BASENODE]:
        """Traverse the graph in topologitcal order."""

        def _in_degree(graph: BaseGraph):
            degree = {}
            for name, node in graph.nodes.items():
                degree[name] = len(node.prev_nodes)
            return degree

        def find_zero_degree_node(in_degree):
            for node_name in in_degree:
                if in_degree[node_name] == 0:
                    return node_name
            raise Exception(f'no zero degree node\n{in_degree}')

        in_degree = _in_degree(self)

        while len(in_degree) > 0:
            node_name = find_zero_degree_node(in_degree)  # visit the node
            in_degree.pop(node_name)
            yield self.nodes[node_name]
            for next in self.nodes[node_name].next_nodes:
                in_degree[next.name] -= 1

    def topo_sort(self):
        """Sort all node in topological order."""
        sorted_nodes = OrderedDict()
        for node in self.topo_traverse():
            sorted_nodes[node.name] = node
        self.nodes = sorted_nodes
