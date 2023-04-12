# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import print_log

from .optim_utils import ONNXOptimUtils

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    from mmrazor.utils import get_package_placeholder
    onnx = get_package_placeholder('No module named onnx')
    numpy_helper = get_package_placeholder('No module named onnx.numpy_helper')

SUPPORT_QWEIGHT_NODE = ['Gemm', 'Conv', 'ConvTranspose']

PERCHANNEL_FAKEQUANTIZER = [
    'FakeQuantizeLearnablePerchannelAffine', 'FixedPerChannelAffine'
]
PERTENSOR_FAKEQUANTIZER = ['LearnablePerTensorAffine', 'FixedPerTensorAffine']

ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER


def _parse_attrs(node_attrs):
    attrs = {}
    for attr in node_attrs:
        if attr.type == onnx.AttributeProto.AttributeType.INTS:
            attrs[attr.name] = tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.AttributeType.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            attrs[attr.name] = tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.AttributeType.STRING:
            attrs[attr.name] = str(attr.s)
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            attrs[attr.name] = tuple([str(x) for x in attr.strings])
        else:
            raise Exception('ATTR Type [{}] Not Supported!'.format(attr.type))
    return attrs


class BaseQuantizeExportor():

    optimizer = ONNXOptimUtils

    def __init__(self, onnx_model, export_path) -> None:

        if isinstance(onnx_model, str):
            self.onnx_model = onnx.load(onnx_model)
        elif isinstance(onnx_model, onnx.ModelProto):
            self.onnx_model = onnx_model
        else:
            raise TypeError

        self.export_path = export_path
        self._init_mappings_from_onnx(self.onnx_model)

        self.optimizer.remove_fake_pad_op(self.onnx_model, self.name2data,
                                          self.input2node, self.output2node)

        self._remap_input_and_node()
        self._remap_output_and_node()

    @property
    def graph(self):
        """The onnx model's graph."""
        return self.onnx_model.graph

    def _init_mappings_from_onnx(self, onnx_model):
        """Build necessary mappings in a onnx model."""

        self.input2node = self.optimizer.map_input_and_node(onnx_model)
        self.output2node = self.optimizer.map_output_and_node(onnx_model)
        self.name2data = self.optimizer.map_name_and_data(onnx_model)

    def _remap_input_and_node(self):
        """Rebuild the mapping from input name to a (node, input index)
        tuple."""
        self.input2node = self.optimizer.map_input_and_node(self.onnx_model)

    def _remap_output_and_node(self):
        """Rebuild the mapping from a node's output name to this node."""
        self.output2node = self.optimizer.map_output_and_node(self.onnx_model)

    def parse_qparams(self, node: onnx.NodeProto):
        """Parse the quantize-related parameters based on a node."""
        tensor_name, scale, zero_point = node.input[:3]

        scale, zero_point = self.name2data[scale], self.name2data[zero_point]
        if len(node.input) > 3:
            qmin, qmax = node.input[-2:]
            qmin, qmax = self.name2data[qmin], self.name2data[qmax]
        elif len(node.attribute) > 0:
            qparams = _parse_attrs(node.attribute)
            qmin = qparams['quant_min']
            qmax = qparams['quant_max']
        else:
            print_log(f'qmin and qmax are not found for <{node.name}>!')
            qmax = qmin = None
        return tensor_name, scale, zero_point, qmin, qmax

    def collect_symbolic_nodes(self, onnx_model: onnx.ModelProto):
        """Collect all the fakequant nodes from a onnx model."""
        symbolic_nodes = list()
        for node in onnx_model.graph.node:
            if node.op_type in ALL_FAKEQUANTIZER:
                symbolic_nodes.append(node)
        return symbolic_nodes

    def _get_constant_inputs(self, node: onnx.NodeProto):
        """Get the constant input node for the current node."""
        constant_nodes = list()
        output2node = self.output2node
        for inp in node.input:
            if inp in output2node and output2node[inp].op_type == 'Constant':
                cnode = output2node[inp]

                constant_nodes.append(cnode)
        return constant_nodes

    def _collect_symbolic_constant_inputs(self, symbolic_nodes: List):
        """Collect these constant nodes which is the input of all the symbolic
        node."""

        collected_constant_names = set()
        constant_inputs = list()
        for node in symbolic_nodes:
            constant_inputs = self._get_constant_inputs(node)
            for constant in constant_inputs:
                if constant.name in collected_constant_names:
                    continue
                constant_inputs.append(constant)
                collected_constant_names.add(constant.name)
        return constant_inputs

    def _remove_symbolic_related_from_onnx(self, symbolic_nodes: List,
                                           symbolic_constant_inputs: List):
        """Remove these out of date fakequant nodes and theirs constant input
        nodes."""
        for node in symbolic_nodes:
            self.onnx_model.graph.node.remove(node)

        # Remove symbolic related constant nodes. The constant node which is
        # only used by those symbolic nodes can be removed.

        def _is_standalone_constant_node(constant):
            for node in self.onnx_model.graph.node:
                for input_name in node.input:
                    # A constant node always has one output.
                    if input_name == constant.output[0]:
                        return False
            return True

        for constant in symbolic_constant_inputs:
            if _is_standalone_constant_node(constant):
                self.onnx_model.graph.node.remove(constant)

    def export(self):
        """Export end to end onnx model."""
        # todo: is it a abstract method?
        raise NotImplementedError
