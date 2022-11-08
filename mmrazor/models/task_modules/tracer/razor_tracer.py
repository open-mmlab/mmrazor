# Copyright (c) OpenMMLab. All rights reserved.
"""This module define FxTracer and related classes."""

from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.node import Node as FxNode

from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph.base_graph import BaseGraph, BaseNode
from .fx_tracer import CostumFxTracer


class FxBaseNode(BaseNode):
    """Node to record FxNode."""

    def __init__(self, name: str, val: FxNode) -> None:
        super().__init__(name, val)

    def module(self):
        """Union[Module | None]: the module the fxnode corresponding to."""
        self.val: FxNode
        model = self.val.graph.owning_module
        if self.val.op == 'call_module':
            target = self.val.target
            target = target.split('.')
            obj = model
            for t in target:
                obj = getattr(obj, t)
            return obj
        else:
            return None

    def function(self):
        """Union[Callable | Node]: the function the fxnode corresponding to."""
        if self.is_function():
            return self.val.target
        else:
            return None

    def method(self):
        if self.is_method():
            return self.val.target
        else:
            return None

    # base type
    # placeholder|call_method|call_module|call_function|get_attr|output

    def is_function(self):
        """Bool: if the fxnode represents 'call_function'"""
        return self.val.op == 'call_function'

    def is_module(self):
        """Bool: if the fxnode represents 'call_module'"""
        return self.val.op == 'call_module'

    def is_input(self):
        """Bool: if the fxnode represents input or output tensors"""
        return self.val.op == 'placeholder'

    def is_output(self):
        return self.val.op == 'output'

    def is_method(self):
        """Bool: if the fxnode represents 'call_method'"""
        return self.val.op == 'call_method'

    def is_get_attr(self):
        """Bool: if the fxnode represents 'get_attr'"""
        return self.val.op == 'get_attr'

    # extended type

    def is_cat(self):
        """Bool: if the fxnode represents a cat node"""
        return self.is_function() and self.function() is torch.cat

    # other

    def __repr__(self) -> str:
        return f'{self.name}({self.val.op})'


@TASK_UTILS.register_module()
class RazorFxTracer(CostumFxTracer):
    """A wapper for torch.fx.tracer."""

    def __init__(self,
                 is_extra_leaf_module: Callable[[nn.Module, str], bool] = None,
                 concrete_args={}) -> None:
        if isinstance(is_extra_leaf_module, dict):
            is_extra_leaf_module = TASK_UTILS.build(is_extra_leaf_module)
        super().__init__(
            is_extra_leaf_module=is_extra_leaf_module,
            concrete_args=concrete_args)

    def add_node(self, graph: BaseGraph[FxBaseNode], fxnode: FxNode):
        """FxBaseNode: convert a torch FxNode to a FxBaseNode, and add it the
        self.graph"""
        node = graph.add_or_find_node(FxBaseNode(fxnode.name, fxnode))
        return node

    def parse_torch_graph(self, torch_graph: fx.graph.Graph):
        """None: convert torch graph to self.graph"""

        graph = BaseGraph[FxBaseNode]()
        # copy_nodes
        for fxnode in torch_graph.nodes:
            self.add_node(graph, fxnode)

        # connect nodes
        for fxnode in torch_graph.nodes:
            for pre_node in fxnode.all_input_nodes:
                graph.connect(
                    self.add_node(graph, pre_node),
                    self.add_node(graph, fxnode))

        return graph

    def trace(self, model) -> BaseGraph[FxBaseNode]:
        torch_graph = super().trace(model)
        torch_graph.owning_module = model

        self.graph = BaseGraph[FxBaseNode]()
        return self.parse_torch_graph(torch_graph)
