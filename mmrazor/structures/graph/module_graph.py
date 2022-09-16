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
from .base_graph import BaseGraph, BaseNode

# ModuleNode && ModuleGraph


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
                 expand_ratio: int = 1,
                 module_name='') -> None:
        """
        Args:
            name (str): the name of the node
            val (Module | str): content of the node. It can be Module or
            string. If val is a string, the string can only be one of
                self.pre_defined_node_val_str
            expand_ratio (int): expand_ratio is used in bind node,
                where the out_channel is always a multiple of the in_channel.
        Note:
            Here, we give an example of expand_ratio.
            >>> class Pool(nn.Module):
                    def forward(x):
                        return F.adaptive_avg_pool2d(x,2).flatten(1)
            >>> node= ModuleNode('pass_0',Pool(),expand_ratio=4)
            >>> assert node.out_channels == node.in_channels*4
        """

        assert (isinstance(val, Module)
                or val in self.__class__.pre_defined_node_val_str
                ), f'{val} node is not allowed'
        if expand_ratio != 1:
            assert val == 'pass_placeholder', \
                'expand != 1 is only valid when val=="pass"'
        super().__init__(name, val)
        self.expand_ratio = expand_ratio
        self.module_name = module_name

    # channel

    @property
    def in_channels(self) -> int:
        """int: the in_channels of the node."""
        if isinstance(self.val, nn.Module):
            MAPPING = {
                nn.Conv2d: 'in_channels',
                nn.modules.batchnorm._BatchNorm: 'num_features',
                nn.modules.Linear: 'in_features',
            }
            for basetype in MAPPING:
                if isinstance(self.val, basetype):
                    return getattr(self.val, MAPPING[basetype])
            raise NotImplementedError(f'unsupported module: {self.val}')
        elif self.is_bind_node() or self.is_pass_node():
            if len(self.prev_nodes) > 0:
                return self.prev_nodes[0].out_channels
            else:
                return 0
        elif self.is_cat_node():
            return sum([
                node.out_channels if node.out_channels is not None else 0
                for node in self.prev_nodes
            ])
        else:
            raise NotImplementedError(
                f'unsupported node type: {self.basic_type}')

    @property
    def out_channels(self) -> int:
        """int: the out_channels of the node."""
        if isinstance(self.val, nn.Module):
            MAPPING = {
                nn.Conv2d: 'out_channels',
                nn.modules.batchnorm._BatchNorm: 'num_features',
                nn.modules.Linear: 'out_features',
            }
            for basetype in MAPPING:
                if isinstance(self.val, basetype):
                    return getattr(self.val, MAPPING[basetype])
            raise NotImplementedError(f'unsupported module: {self.val}')
        elif self.is_bind_node():
            if len(self.prev_nodes) > 0:
                return self.prev_nodes[0].out_channels
            else:
                return 0
        elif self.is_pass_node():
            return self.in_channels * self.expand_ratio
        elif self.is_cat_node():
            return sum([
                node.out_channels if node.out_channels is not None else 0
                for node in self.prev_nodes
            ])
        else:
            raise NotImplementedError(
                f'unsupported node type: {self.basic_type}')

    # other

    def __repr__(self) -> str:
        return f'{self.name}_({self.in_channels},{self.out_channels})'

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
                raise NotImplementedError(f'{self}')
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

    # check

    def check_channel(self):
        """Check if the channels of the node is matchable with previous nodes
        and next nodes."""
        if self.is_cat_node():
            pass
        else:
            for pre in self.prev_nodes:
                assert pre.out_channels == self.in_channels, \
                    f'{self} has channel error'

    def check_type(self):
        """Check if the node has right number of previous nodes according to
        their type."""
        if self.is_pass_node():
            assert len(self.prev_nodes) <= 1, '{name} pass node error'
        elif self.is_cat_node():
            pass
        elif self.is_bind_node():
            assert len(self.prev_nodes) > 1, '{name} bind node error'
        elif self.is_mix_node():
            assert len(self.prev_nodes) <= 1, '{name} mix node error'
        else:
            raise NotImplementedError(f'{self}')


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
    def init_from_fx_tracer(model: Module,
                            fx_tracer={'type': 'RazorFxTracer'}):
        """init module graph using torch fx tracer."""
        pass

    @staticmethod
    def init_from_model(model: Module):
        """init module graph from a model which uses connect_module to record
        the relation among modules."""
        pass

    # check

    def check(self):
        """Check if the graph is valid."""
        for node in self:
            node.check_channel()
            node.check_type()

    # static method for models that can't use tracer

    @staticmethod
    def connect_module(pre: Module, next: Module):
        """This function is used to write hardcode in modules to generate Graph
        object using init_from_model."""
        if hasattr(pre, '_next'):
            _next = getattr(pre, '_next')
            assert isinstance(_next, List)
        else:
            pre._next = set()
        pre._next.add(next)

        if hasattr(next, '_pre'):
            _pre = getattr(next, '_pre')
            assert isinstance(_pre, List)
        else:
            next._pre = set()
        next._pre.add(pre)

    # others
    def refresh_module_name(self):
        module2name = {}
        for name, module in self._model.named_modules():
            module2name[module] = name

        for node in self:
            if isinstance(node.val, nn.Module):
                node.module_name = module2name[node.val]


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
        node = ModuleNode(f'{type}_{num}', type, expand_ratio=expand_ratio)
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
                    assert node.in_channels % pre.out_channels == 0
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
        self._remove_redundant_pass_nodes()
        self._insert_bind_nodes()
        self._insert_pass_nodes()
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

        self._post_process()

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
