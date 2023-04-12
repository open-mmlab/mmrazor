# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

from mmengine import print_log

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    from mmrazor.utils import get_package_placeholder
    onnx = get_package_placeholder('No module named onnx')
    numpy_helper = get_package_placeholder('No module named onnx.numpy_helper')


class ONNXOptimUtils():

    @classmethod
    def map_name_and_data(cls, onnx_model: onnx.ModelProto):
        """Build the mapping from a data's name to the data itself."""
        params = {}
        for init in onnx_model.graph.initializer:
            params[init.name] = numpy_helper.to_array(init)
        for node in onnx_model.graph.node:
            # If two zero_points are identity, one is a reference to the other
            # after optimized by onnx.
            if node.op_type == 'Identity' and len(node.input) == 1 and \
                    node.input[0] in params:
                params[node.output[0]] = copy.deepcopy(params[node.input[0]])
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        params[node.output[0]] = numpy_helper.to_array(attr.t)
        return params

    @classmethod
    def map_name_and_initializer(cls,
                                 onnx_model: onnx.ModelProto,
                                 allow_redundant=True):
        """Build the mapping from a initializer's output name to this
        initializer."""

        initializers = dict()

        for idx, init in enumerate(onnx_model.graph.initializer):
            initializers[init.name] = (init, idx)

        return initializers

    @classmethod
    def map_output_and_node(cls, onnx_model: onnx.ModelProto):
        """Build the mapping from a node's output name to this node."""
        output2node = dict()
        for node in onnx_model.graph.node:
            for output_name in node.output:
                output2node[output_name] = node
        return output2node

    @classmethod
    def map_input_and_node(cls, onnx_model: onnx.ModelProto):
        """Build the mapping from input name to a (node, input index) tuple."""

        input2node: Dict[str, List] = dict()
        for node in onnx_model.graph.node:
            for idx, input_name in enumerate(node.input):
                if input_name not in input2node:
                    input2node[input_name] = []
                input2node[input_name].append([node, idx])
        return input2node

    @classmethod
    def remove_node_from_onnx(cls, node: onnx.NodeProto,
                              onnx_model: onnx.ModelProto):
        """Removes a node from node list."""
        onnx_model.graph.node.remove(node)

    @classmethod
    def remove_initializer_from_onnx(cls, initializer: onnx.TensorProto,
                                     onnx_model: onnx.ModelProto):
        """Inserts the initializer at the specified position."""
        onnx_model.graph.initializer.remove(initializer)

    @classmethod
    def remove_fake_pad_op(cls, onnx_model, name2data, inp2node, out2node):
        nodes_to_be_removed = []
        for idx, node in enumerate(onnx_model.graph.node):
            if node.op_type == 'Pad':
                pads = name2data[node.input[1]]
                if all([x == 0 for x in pads]):
                    print_log(f'Remove pad op: <{node.name}>.')
                    next_nodes = inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]
                    nodes_to_be_removed.append(node)

        for node in nodes_to_be_removed:
            onnx_model.graph.node.remove(node)

    @classmethod
    def insert_node_to_onnx(cls,
                            node: onnx.NodeProto,
                            onnx_model: onnx.ModelProto,
                            idx: int = 0):
        """Inserts the node at the specified position."""
        onnx_model.graph.node.insert(idx, node)

    @classmethod
    def find_standalone_nodes(cls,
                              onnx_model: onnx.ModelProto,
                              input2node: Optional[Dict] = None,
                              output2node: Optional[Dict] = None):
        """Find unused nodes."""

        if input2node is None:
            input2node = cls.map_input_and_node(onnx_model)
        if output2node is None:
            output2node = cls.map_output_and_node(onnx_model)

        def _is_standalone_node(node, input2node, output2node):
            for input_name in node.input:
                if input_name in output2node:
                    return False

            for out_node in node.output:
                if out_node in input2node:
                    return False

            return True

        standalone_nodes = list()
        for node in onnx_model.graph.node:

            if _is_standalone_node(node, input2node, output2node):
                standalone_nodes.append(node)
        return standalone_nodes

    @classmethod
    def find_redundant_initializers(cls,
                                    onnx_model: onnx.ModelProto,
                                    input2node: Optional[Dict] = None):
        """Find unused initializers."""
        if input2node is None:
            input2node = cls.map_input_and_node(onnx_model)

        initializers = cls.map_name_and_initializer(onnx_model)
        redundant_initializers = list()
        redundant_set = set()
        for name, init_and_idx in initializers.items():
            if name not in input2node and name not in redundant_set:
                # init_and_idx[0] is onnx.onnx_ml_pb2.TensorProto
                # init_and_idx[1] is a integer index
                redundant_initializers.append(init_and_idx[0])
                redundant_set.add(name)
        return redundant_initializers

    @classmethod
    def topo_sort(cls,
                  onnx_model: onnx.ModelProto,
                  initializers: Optional[Dict] = None,
                  inplace: bool = True):
        """Topologically sort the nodes in a directed acyclic graph.

        Note that nodes in a directed acyclic graph may be out of order
        after replacing symbolic related nodes with new nodes.

        Args:
            onnx_model (onnx.ModelProto): The onnx model to be sorted
                topologically.
            initializers (Dict | Optional): The mapping from name to
                initializers. Default to None.
            inplace (bool): Can optionally do the operation in-place.
                Defaults to True.
        """

        if inplace:
            _onnx_model = onnx_model
        else:
            _onnx_model = copy.deepcopy(onnx_model)

        if initializers is None:
            initializers = cls.map_name_and_initializer(
                _onnx_model, allow_redundant=True)

        # A node may have multiple outputs. The first output name of a node
        # named `/conv/Conv` is `/conv/Conv_output_0`
        output_name2node = {}
        for node in _onnx_model.graph.node:
            for output_name in node.output:
                output_name2node[output_name] = node
        for node in _onnx_model.graph.input:
            output_name2node[node.name] = node

        name2node = {node.name: node for node in _onnx_model.graph.node}

        graph: Dict[str,
                    List] = {node.name: []
                             for node in _onnx_model.graph.node}
        for node in _onnx_model.graph.input:
            graph[node.name] = []

        indegree = {node.name: 0 for node in _onnx_model.graph.node}

        # Build graph
        for i, node in enumerate(_onnx_model.graph.node):
            for input_name in node.input:
                if input_name not in initializers:
                    indegree[node.name] += 1
                    prev_node = output_name2node[input_name]
                    graph[prev_node.name].append(node)

        graph_input = [node.name for node in _onnx_model.graph.input]
        root = graph_input.copy()
        sorted_nodes = []

        # There are some nodes whose input are all initializers.
        for node_name, in_degree in indegree.items():
            if in_degree == 0:
                root.append(node_name)

        while root:
            node_name = root.pop()
            # There is no intersection between graph_input and
            # _onnx_model.graph.node
            if node_name not in graph_input:
                node = name2node[node_name]
                sorted_nodes.append(node)
            for next_node in graph[node_name]:
                indegree[next_node.name] -= 1
                if indegree[next_node.name] == 0:
                    root.append(next_node.name)

        num_nodes = len(_onnx_model.graph.node)
        if len(sorted_nodes) != num_nodes:
            raise RuntimeError('The graph is not a DAG.')

        for _ in range(num_nodes):
            _onnx_model.graph.node.pop()
        for node in sorted_nodes:
            _onnx_model.graph.node.append(node)

        return _onnx_model

    @classmethod
    def optimize(cls, onnx_model):
        """Remove standalone nodes and redundant initializers, and
        topologically sort the nodes in a directed acyclic graph."""

        input2node = cls.map_input_and_node(onnx_model)
        output2node = cls.map_output_and_node(onnx_model)

        standalone_nodes = cls.find_standalone_nodes(onnx_model, input2node,
                                                     output2node)
        for node in standalone_nodes:
            cls.remove_node_from_onnx(node, onnx_model)
            print_log(f'Remove node {node.name}')

        redundant_inits = cls.find_redundant_initializers(
            onnx_model, input2node)
        for init in redundant_inits:
            cls.remove_initializer_from_onnx(init, onnx_model)
            print_log(f'Remove initializer {init.name}')

        sorted_onnx_model = cls.topo_sort(onnx_model)

        return sorted_onnx_model
