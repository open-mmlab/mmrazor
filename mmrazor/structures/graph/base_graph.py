# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Generic, Iterator, List, TypeVar

# BaseNode && BaseGraph


class BaseNode:

    def __init__(self, name: str, val: Any) -> None:
        self.val = val
        self.name = name
        self.pre: List = []
        self.next: List = []

    # node operation

    def add_pre(self, node: 'BaseNode'):
        """add previous node."""
        if node not in self.pre:
            self.pre.append(node)
        if self not in node.next:
            node.next.append(self)

    def add_next(self, node: 'BaseNode'):
        """add next node."""
        if node not in self.next:
            self.next.append(node)
        if self not in node.pre:
            node.pre.append(self)

    # compare operation

    def __hash__(self) -> int:
        return hash((self.val, self.name))

    def __eq__(self, other):
        return self.val is other.val and self.name == other.name

    # other

    def __repr__(self) -> str:
        return self.name


BASENODE = TypeVar('BASENODE', bound=BaseNode)


class BaseGraph(Generic[BASENODE]):

    def __init__(self) -> None:
        super().__init__()
        self.nodes: OrderedDict[str, BASENODE] = OrderedDict()

    # graph operations

    @staticmethod
    def default_node_converter(node: BaseNode):
        return BaseNode(node.name, node.val)

    @classmethod
    def copy_from(cls,
                  graph: 'BaseGraph',
                  node_converter=default_node_converter):
        """This function is used to copy a graph to a new graph of the current
        class.

        node_converter can be used to convert the type of nodes.
        """
        old2new = {}
        new_graph = cls()
        # copy nodes
        for old in graph:
            old2new[old] = new_graph.add_or_find_node(node_converter(old))

        # connect
        for old in graph:
            for pre in old.pre:
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
        """Record a Node."""
        if node.name not in self.nodes:
            self.nodes[node.name] = node
        else:
            raise BaseException(f'{node.name} already exists in graph')

    def connect(self, pre_node: BASENODE, next_node: BASENODE):
        """Add an edge from pre_node to next_node."""
        assert pre_node in self and next_node in self
        pre_node.add_next(next_node)
        next_node.add_pre(pre_node)

    def disconnect(self, pre_node: BASENODE, next_node: BASENODE):
        """Remove the edge form pre_node to next_node."""
        assert pre_node in self and next_node in self
        if next_node in pre_node.next:
            pre_node.next.remove(next_node)
        if pre_node in next_node.pre:
            next_node.pre.remove(pre_node)

    def delete_node(self, node: BASENODE):
        """Delete a node with its related edges."""
        pass

    # work as a collection

    def __iter__(self) -> Iterator[BASENODE]:
        for x in self.nodes.values():
            yield x

    def __contains__(self, node: BASENODE):
        return node.name in self.nodes

    def __len__(self):
        return len(self.nodes)

    # other

    def __repr__(self):
        res = f'Graph with {len(self)} nodes:\n'
        for node in self:
            res += '{0:<40} -> {1:^40} -> {2:<40}\n'.format(
                str(node.pre), node.__repr__(), str(node.next))
        return res

    # traverse

    def topo_traverse(self) -> Iterator[BASENODE]:
        """Traverse the graph in topologitcal order."""

        def _in_degree(graph: BaseGraph):
            degree = {}
            for name, node in graph.nodes.items():
                degree[name] = len(node.pre)
            return degree

        def find_zero(in_degree):
            for node_name in in_degree:
                if in_degree[node_name] == 0:
                    return node_name
            return None

        in_degree = _in_degree(self)

        while len(in_degree) > 0:
            node_name = find_zero(in_degree)  # visit the node
            in_degree.pop(node_name)
            yield self.nodes[node_name]
            for next in self.nodes[node_name].next:
                in_degree[next.name] -= 1

    def topo_sort(self):
        """Sort all node in topological order."""
        sorted_nodes = OrderedDict()
        for node in self.topo_traverse():
            sorted_nodes[node.name] = node
        self.nodes = sorted_nodes
