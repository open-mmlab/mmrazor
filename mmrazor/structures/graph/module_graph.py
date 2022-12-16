# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines ModuleNode and ModuleGraph.

They model the computation graph of a model based on BaseNode and BaseGraph
"""
import copy
from collections import OrderedDict
from typing import Dict, List, TypeVar, Union

import torch.nn as nn
from torch.nn import Module

from mmrazor.models.task_modules.tracer.backward_tracer import BackwardTracer
from mmrazor.models.task_modules.tracer.loss_calculator import \
    ImageClassifierPseudoLoss
from mmrazor.models.task_modules.tracer.path import (Path, PathConcatNode,
                                                     PathList, PathNode)
from mmrazor.registry import TASK_UTILS
from mmrazor.utils import print_log
from .base_graph import BaseGraph, BaseNode
from .pseudo_fx_graph import FxBaseNode


# ModuleNode && ModuleGraph
class NoOutputError(Exception):
    """An error occurs when no output node for a leaf node."""

    def __init__(self, node, *args: object) -> None:
        super().__init__(f'{node}', *args)
        self.node = node

    pass


class NoInputError(Exception):
    """An error occurs when no input node for a leaf node."""

    def __init__(self, node, *args: object) -> None:
        super().__init__(f'{node}', *args)
        self.node = node


def my_assert(condiion, exception):
    """assert helper function."""
    if not condiion:
        raise exception


class ModuleNode(BaseNode):
    """A node in a computation graph.

    All nodes are divided to four types, the detail of definition can be found
    in functions  self.is_{xxx}_node.
    """

    pre_defined_node_val_str = [
        'cat_placeholder', 'bind_placeholder', 'pass_placeholder'
    ]

    def __init__(self,
                 name: str,
                 val: Union[Module, str],
                 module_name='') -> None:
        """
        Args:
            name (str): the name of the node
            val (Module | str): content of the node. It can be Module or
            string. If val is a string, the string can only be one of
                self.pre_defined_node_val_str
        Note:
            Here, we give an example of expand_ratio.
            >>> class Pool(nn.Module):
                    def forward(x):
                        return F.adaptive_avg_pool2d(x,2).flatten(1)
            >>> node= ModuleNode('pass_0',Pool(),expand_ratio=4)
            >>> assert node.out_channels == node.in_channels*4
        """

        super().__init__(name, val)
        self.module_name = module_name

    # other

    @property
    def is_module(self):
        """Whether the node includes a module."""
        return isinstance(self.val, nn.Module)

    def __repr__(self) -> str:
        repr = f'{self.name}'
        if self.module_name != '':
            repr += f'({self.module_name})'
        return repr

    # node type

    @property
    def basic_type(self) -> str:
        """The basic type of the node.

        Basic types are divided into seveval major types, detailed in
        self.is_{xxx}_node
        """
        if isinstance(self.val, Module):
            if isinstance(self.val, nn.Conv2d):
                if self.val.groups == 1:
                    return 'conv2d'
                elif self.val.groups == self.val.in_channels == \
                        self.val.out_channels:
                    return 'dwconv2d'
                else:
                    return 'gwconv2d'
            elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                return 'bn'
            elif isinstance(self.val, nn.Linear):
                return 'linear'
            else:
                raise NotImplementedError(f'{self.val}')
        else:
            if self.val in [
                    'cat_placeholder', 'bind_placeholder', 'pass_placeholder'
            ]:
                return self.val
            else:
                raise NotImplementedError()

    def is_pass_node(self):
        """pass node represent a module whose in-channels correspond out-
        channels one-to-one."""
        return self.basic_type in ['bn', 'dwconv2d', 'pass_placeholder']

    def is_cat_node(self):
        """cat node represents a cat module."""
        return self.basic_type == 'cat_placeholder'

    def is_bind_node(self):
        """bind node represent a node that has multiple inputs, and their
        channels are bound one-to-one."""
        return self.basic_type == 'bind_placeholder'

    def is_mix_node(self):
        """mix node represents a module that mixs all input channels and
        generete new output channels, such as conv and linear."""
        return self.basic_type in ['conv2d', 'linear', 'gwconv2d']

    def is_input(self):
        """Whether the node is an input node."""
        return self.val == 'input_placeholder'

    def is_output(self):
        """Whether the node is an output node."""
        return self.val == 'output_placeholder'

    def check(self):
        """Check whether the node has any error."""
        if self.is_input():
            assert len(self.prev_nodes) == 0, f'{self}'
            my_assert(len(self.next_nodes) > 0, NoOutputError(self))
        elif self.is_output():
            my_assert(len(self.prev_nodes) > 0, NoInputError(self))
            assert len(self.next_nodes) == 0, f'{self}'
        else:
            my_assert(len(self.prev_nodes) > 0, NoInputError(self))
            my_assert(len(self.next_nodes) > 0, NoOutputError(self))


MODULENODE = TypeVar('MODULENODE', bound=ModuleNode)


class ModuleGraph(BaseGraph[MODULENODE]):
    """Computatation Graph."""

    def __init__(self, model=None) -> None:
        super().__init__()
        self._model: nn.Module = model

    # functions to generate module graph.

    @staticmethod
    def init_from_backward_tracer(
        model: Module,
        backward_tracer=BackwardTracer(
            loss_calculator=ImageClassifierPseudoLoss()),
    ):
        """init module graph using backward tracer."""
        if isinstance(backward_tracer, dict):
            backward_tracer = TASK_UTILS.build(backward_tracer)
        path_lists = backward_tracer.trace(model)
        converter = PathToGraphConverter(path_lists, model)
        converter.graph.refresh_module_name()
        return converter.graph

    @staticmethod
    def init_from_model(model: Module):
        """init module graph from a model which uses connect_module to record
        the relation among modules."""
        pass

    # others
    def refresh_module_name(self):
        """Refresh the module name."""
        module2name = {}
        for name, module in self._model.named_modules():
            module2name[module] = name

        for node in self:
            if isinstance(node.val, nn.Module):
                node.module_name = module2name[node.val]

    def check(self, fix=False):
        """Check whether the Graph has any error."""
        for node in copy.copy(list(self.topo_traverse())):
            self._check(node, fix=fix)

    def _check(self, node, fix=False):
        """Helper method for self.check."""
        try:
            node.check()
        except Exception as e:
            if not fix:
                raise e
            else:
                try:
                    raise e
                except NoOutputError as e:
                    print_log(
                        f'add a output after {node}, error: {e}',
                        level='debug')
                    self._add_output_after(node)
                except NoInputError as e:
                    print_log(
                        f'add a input before {node}, error: {e}',
                        level='debug')
                    self._add_input_before(node)

                self._check(node, fix=True)

    def _add_input_before(self, node):
        """Add an input node before a node."""
        input_node = ModuleNode('auto_input',
                                'input_placeholder')  # type: ignore
        input_node = self.add_or_find_node(input_node)
        self.connect(input_node, node)

    def _add_output_after(self, node):
        """Add an output node after a node."""
        output_node = ModuleNode('auto_output',
                                 'output_placeholder')  # type: ignore
        output_node = self.add_or_find_node(output_node)
        self.connect(node, output_node)


# Converter


class GraphConverter:
    """Base class for converters for ModuleGraph."""

    def __init__(self, model) -> None:
        self.graph = ModuleGraph[ModuleNode](model)
        self.cat_placeholder_num = 0
        self.bind_placeholder_num = 0
        self.pass_placeholder_num = 0

    # add node

    def _new_placeholder_node(self, type: str, expand_ratio=1):
        """New cat/bind/pass node."""
        assert type in [
            'cat_placeholder', 'pass_placeholder', 'bind_placeholder'
        ]
        if expand_ratio != 1:
            assert type == 'pass_placeholder'
        if type == 'cat_placeholder':
            num = self.cat_placeholder_num
            self.cat_placeholder_num += 1
        elif type == 'pass_placeholder':
            num = self.pass_placeholder_num
            self.pass_placeholder_num += 1
        elif type == 'bind_placeholder':
            num = self.bind_placeholder_num
            self.bind_placeholder_num += 1
        else:
            pass
        node = ModuleNode(f'{type}_{num}', type)
        self.graph.add_or_find_node(node)
        return node

    # insert nodes

    def _insert_node_before(self, node: ModuleNode, new_node: ModuleNode):
        """Insert a new node before a node."""
        for pre in node.prev_nodes:
            self.graph.connect(pre, new_node)
        for pre in new_node.prev_nodes:
            self.graph.disconnect(pre, node)
        self.graph.connect(new_node, node)

    def _insert_bind_nodes(self):
        """Add bind nodes before the nodes which only need one previous node
        but have more than one."""

        need_bind_nodes = []
        for node in self.graph:
            if (isinstance(node.val, nn.Conv2d)
                    or isinstance(node.val, nn.Linear)
                    or isinstance(node.val, nn.modules.batchnorm._BatchNorm)):
                if len(node.prev_nodes) > 1:
                    need_bind_nodes.append(node)
        for node in need_bind_nodes:
            bind_node = self._new_placeholder_node('bind_placeholder')
            self._insert_node_before(node, bind_node)

    def _insert_pass_nodes(self):
        """Add pass nodes where the channel conflict."""
        for node in copy.copy(list(self.graph.nodes.values())):
            if len(node.prev_nodes) == 1:
                pre: ModuleNode = node.prev_nodes[0]
                if node.in_channels != pre.out_channels:
                    assert node.in_channels % pre.out_channels == 0, \
                        f'{node.name} channel error'
                    pass_node = self._new_placeholder_node(
                        'pass_placeholder',
                        node.in_channels // pre.out_channels)
                    self._insert_node_before(node, pass_node)

    def _remove_redundant_pass_nodes(self):
        """Remove redundant pass nodes, which do not change number of channels
        and  do not represent any module."""
        for node in copy.copy(list(self.graph.nodes.values())):
            if (node.is_pass_node() and len(node.prev_nodes) == 1
                    and len(node.next_nodes) == 1
                    and not isinstance(node.val, nn.Module)
                    and node.in_channels == node.out_channels):
                self.graph.delete_node(node)

    # topo_rename_nodes
    def _topo_rename(self):
        """Rename cat, bind, pass nodes in topological order."""
        self.cat_placeholder_num = 0
        self.bind_placeholder_num = 0
        self.pass_placeholder_num = 0
        sorted_nodes = OrderedDict()
        for node in self.graph.topo_traverse():
            node: ModuleNode
            if isinstance(node.val, Module):
                pass
            elif node.is_pass_node():
                node.name = f'pass_{self.pass_placeholder_num}'
                self.pass_placeholder_num += 1
            elif node.is_cat_node():
                node.name = f'cat_{self.cat_placeholder_num}'
                self.cat_placeholder_num += 1
            elif node.is_bind_node():
                node.name = f'bind_{self.bind_placeholder_num}'
                self.bind_placeholder_num += 1
            else:
                pass
            sorted_nodes[node.name] = node
        self.graph.nodes = sorted_nodes

    # other
    def _post_process(self):
        """Some post process after init a basic module graph."""
        # self._remove_redundant_pass_nodes()
        self._insert_bind_nodes()
        self._topo_rename()


class PathToGraphConverter(GraphConverter):
    """The class converts pathlist, which is generated by backward tracer, to a
    module graph."""

    def __init__(self, path_list: PathList, model: Module) -> None:
        """
            Args:
                path_list (PathList): path_list generated by backward tracer.
                model (Module): the model corresponding to the path_list
        """
        super().__init__(model)
        self.path_list = path_list
        self.cat_dict: Dict[str, str] = {}
        self.name2module = dict(model.named_modules())
        self._parse(self.path_list)

        self._insert_bind_nodes()
        self._topo_rename()

    def _parse(self, path_list: PathList):
        """Parse path list."""
        self._parse_helper(path_list, [])

    def _parse_helper(self, path_unit: Union[PathList, Path, PathNode],
                      next_nodes: List[ModuleNode]):
        """Parse a node(unit) in path list."""
        current_node = None
        # path_list
        if isinstance(path_unit, PathList):
            for single_path in path_unit:  # sibling
                self._parse_helper(single_path, next_nodes)

        # path:
        elif isinstance(path_unit, Path):
            current_nexts = next_nodes
            for node in path_unit:  # parent -> children
                current_node = self._parse_helper(node, current_nexts)
                current_nexts = [current_node]

        # Node
        elif isinstance(path_unit, PathNode):

            # cat node: [cat_path_lists]
            if isinstance(path_unit, PathConcatNode):
                current_node = self._add_or_find_node(path_unit)
                self._connect_nexts(current_node, next_nodes)
                for catpath in path_unit.path_lists:  # sibling
                    self._parse_helper(catpath, [current_node])

            # single node
            else:
                current_node = self._add_or_find_node(path_unit)
                self._connect_nexts(current_node, next_nodes)
        return current_node

    def _add_or_find_cat_node(self, pathnode: PathConcatNode):
        """Receive a cat-node.

        If the cat-node exists in the graph, the corresponding node is
        returned, or a new cat node is added to the graph.
        """

        def unify_cat_name(name: str):
            cat_name = name.split('_')
            inputs = sorted(cat_name[1:])
            return f"cat_{'_'.join(inputs)}"

        name_id = pathnode.name
        name_id = unify_cat_name(name_id)
        if name_id in self.cat_dict:
            name = self.cat_dict[name_id]
        else:
            name = f'cat_{self.cat_placeholder_num}'
            self.cat_placeholder_num += 1
            self.cat_dict[name_id] = name
        node = self.graph.add_or_find_node(ModuleNode(name, 'cat_placeholder'))
        return node

    def _add_or_find_node(self, pathnode: PathNode) -> Module:
        """Receive a cat-node.

        If the cat-node exists in the graph, the corresponding node is
        returned, or a new cat node is added to the graph.
        """
        if isinstance(pathnode, PathConcatNode):
            return self._add_or_find_cat_node(pathnode)
        else:
            name = pathnode.name
            assert name in self.name2module, f"{name} doesn't exist in model"
            module = self.name2module[name]
            return self.graph.add_or_find_node(ModuleNode(name, module))

    def _connect_nexts(self, node, nexts: List[ModuleNode]):
        """Connext the node and the nodes in nexts."""
        for next in nexts:
            self.graph.connect(node, next)


class FxTracerToGraphConverter(GraphConverter):
    """Use fx tracer to parse model, and generate module-graph."""

    def __init__(self, base_graph, model=None) -> None:
        """
        Args:
            model (Module): the model which will be parsed
            is_extra_leaf_module (Callable): a function used to determine,
             if a module is a leaf module except torch pre-defined modules
        """
        super().__init__(model)
        self.base_graph = base_graph
        self._convert_graph()

    def _node_converter(self, node: FxBaseNode):
        """Convert a fxnode to a module-node."""
        if node.is_function():
            val = node.function()
        elif node.is_input():
            val = 'input_placeholder'
        elif node.is_output():
            val = 'output_placeholder'
        elif node.is_method():
            val = node.method()
        elif node.is_get_attr():
            val = 'get_attr'
        elif node.is_module():
            val = node.module()
        else:
            raise NotImplementedError(f'{node} is unsupported')

        new_node = ModuleNode(node.name, val)
        return new_node

    def _convert_graph(self):
        """Convert a torch-graph to a module-graph."""
        base_graph = self.base_graph
        # copy_nodes and connect
        module_graph = ModuleGraph.copy_from(base_graph, self._node_converter)
        self.graph = module_graph
