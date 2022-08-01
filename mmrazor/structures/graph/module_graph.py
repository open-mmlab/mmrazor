# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Dict, List, TypeVar, Union

import torch.nn as nn
from torch.nn import Module

from ..tracer.backward_tracer import BackwardTracer
from ..tracer.loss_calculator import ImageClassifierPseudoLoss
from ..tracer.path import ConcatNode as PathCatNode
from ..tracer.path import Node as PathNode
from ..tracer.path import Path, PathList
from .base_graph import BaseGraph, BaseNode

# ModuleNode && ModuleGraph


class ModuleNode(BaseNode):
    pre_defined_node_val = ['cat', 'bind', 'pass']

    def __init__(self,
                 name: str,
                 val: Union[Module, str],
                 expand_ratio=1) -> None:
        """expand_ratio is used in bind-node where the out_channel is always a
        multiple of the in_channel.

        Example:
            y=AdaptiveAvgPooling(size=2)(x).flatten(1)
            channel(x)=4*channel(y)
        """
        assert isinstance(
            val, Module
        ) or val in self.__class__.pre_defined_node_val, \
            f'{val} node is not allowed'
        super().__init__(name, val)
        assert expand_ratio == 1 or val == 'pass'
        self.expand_ratio = expand_ratio

    # channel

    @property
    def in_channels(self):
        """return the in_channels of the node."""
        if isinstance(self.val, nn.Module):
            if isinstance(self.val, nn.Conv2d):
                return self.val.in_channels
            elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                return self.val.num_features
            elif isinstance(self.val, nn.Linear):
                return self.val.in_features
            else:
                raise NotImplementedError(f'unsupported module: {self.val}')
        elif self.is_bind_node() or self.is_pass_node():
            if len(self.pre) > 0:
                return self.pre[0].out_channels
            else:
                return 0
        elif self.is_cat_node():
            return sum([
                node.out_channels if node.out_channels is not None else 0
                for node in self.pre
            ])
        else:
            raise NotImplementedError(f'unsupported node type: {self.type()}')

    @property
    def out_channels(self):
        """return the out_channels of the node."""
        if isinstance(self.val, nn.Module):

            if isinstance(self.val, nn.Conv2d):
                return self.val.out_channels
            elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                return self.val.num_features
            elif isinstance(self.val, nn.Linear):
                return self.val.out_features
            else:
                raise NotImplementedError(f'unsupported module: {self.val}')
        elif self.is_bind_node():
            if len(self.pre) > 0:
                return self.pre[0].out_channels
            else:
                return 0
        elif self.is_pass_node():
            return self.in_channels * self.expand_ratio
        elif self.is_cat_node():
            return sum([
                node.out_channels if node.out_channels is not None else 0
                for node in self.pre
            ])
        else:
            raise NotImplementedError(f'unsupported node type: {self.type()}')

    def check_channel(self):
        """Check if the channels of the node is matchable with previous nodes
        and next nodes."""
        if self.is_cat_node():
            pass
        else:
            for pre in self.pre:
                assert pre.out_channels == self.in_channels, \
                    f'{self} has channel error'

    # other

    def __repr__(self) -> str:
        return f'{self.name}_({self.in_channels},{self.out_channels})'

    # node type

    def type(self) -> str:
        if isinstance(self.val, Module):
            if isinstance(self.val, nn.Conv2d):
                if self.val.groups == 1:
                    return 'conv'
                elif self.val.groups == self.val.in_channels \
                        == self.val.out_channels:
                    return 'dwconv'
                else:
                    raise NotImplementedError('group_wise conv')
            elif isinstance(self.val, nn.modules.batchnorm._BatchNorm):
                return 'bn'
            elif isinstance(self.val, nn.Linear):
                return 'linear'
            else:
                raise NotImplementedError(f'{self}')
        else:
            if self.val in ['cat', 'bind', 'pass']:
                return self.val
            else:
                raise NotImplementedError()

    def is_pass_node(self):
        """pass node represent a module whose in-channels correspond out-
        channels one-to-one."""
        return self.type() in ['bn', 'dwconv', 'pass']

    def is_cat_node(self):
        """cat node represents a cat module."""
        return self.type() == 'cat'

    def is_bind_node(self):
        """bind node represent a node that has multiple inputs, and their
        channels are bound one-to-one."""
        return self.type() == 'bind'

    def is_mix_node(self):
        """mix node represents a module that mixs all input channels and
        generete new output channels."""
        return self.type() in ['conv', 'linear']


MODULENODE = TypeVar('MODULENODE', bound=ModuleNode)


class ModuleGraph(BaseGraph[MODULENODE]):

    # functions to generate module graph.

    @staticmethod
    def init_from_path_list(path_list: PathList, model: Module):
        """init module graph using path lists which are generated by backward
        tracer."""
        converter = PathToGraph(path_list, model)
        return converter.graph

    @staticmethod
    def init_using_backward_tracer(
        model: Module,
        backward_tracer=BackwardTracer(
            loss_calculator=ImageClassifierPseudoLoss())):
        """init module graph using backward tracer."""
        path_lists = backward_tracer.trace(model)
        graph = ModuleGraph.init_from_path_list(path_lists, model)
        return graph

    @staticmethod
    def init_using_fx_tracer(model: Module, is_extra_leaf_module=None):
        """init module graph using torch fx tracer."""
        pass

    @staticmethod
    def init_from_model(model: Module):
        """init module graph from a model which uses connect_module to record
        the relation among modules."""
        pass

    def check_channels(self):
        """check if the channels is matchable among all nodes in the graph."""
        for node in self:
            node.check_channel()

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


# Converter


class ToGraph:

    def __init__(self) -> None:
        self.graph = ModuleGraph[ModuleNode]()
        self.cat_num = 0
        self.bind_num = 0
        self.pass_num = 0

    def new_cat(self):
        node = self.graph.add_or_find_node(
            ModuleNode(f'cat_{self.cat_num}', 'cat'))
        self.cat_num += 1
        return node

    def new_pass(self, expand_ratio=1):
        node = self.graph.add_or_find_node(
            ModuleNode(
                f'pass_{self.pass_num}', 'pass', expand_ratio=expand_ratio))
        self.pass_num += 1
        return node

    def new_bind(self):
        node = self.graph.add_or_find_node(
            ModuleNode(f'bind_{self.bind_num}', 'bind'))
        self.bind_num += 1
        return node

    # add nodes

    def add_bind_nodes(self):
        """deal with special occation."""

        need_bind_nodes = []
        for node in self.graph:
            if isinstance(node.val, nn.Conv2d) \
                    or isinstance(node.val, nn.Linear) \
                    or isinstance(node.val,
                                  nn.modules.batchnorm._BatchNorm):
                if len(node.pre) > 1:
                    need_bind_nodes.append(node)
        for node in need_bind_nodes:
            self.add_bind_node_before(node)

    def add_bind_node_before(self, node: ModuleNode):
        bind_node = self.new_bind()
        for pre in node.pre:
            self.graph.connect(pre, bind_node)
        for pre in bind_node.pre:
            self.graph.disconnect(pre, node)
        self.graph.connect(bind_node, node)

    def add_pass_nodes(self):
        for node in copy.copy(list(self.graph.nodes.values())):
            if len(node.pre) == 1:
                pre: ModuleNode = node.pre[0]
                if node.in_channels != pre.out_channels:
                    assert node.in_channels % pre.out_channels == 0
                    self.add_pass_node_before(node)

    def add_pass_node_before(self, node: ModuleNode):
        assert len(node.pre) == 1
        pre: ModuleNode = node.pre[0]
        assert node.in_channels % pre.out_channels == 0

        pass_node = self.new_pass(node.in_channels // pre.out_channels)

        self.graph.connect(pass_node, node)
        self.graph.connect(pre, pass_node)
        self.graph.disconnect(pre, node)

    # topo_rename_nodes
    def topo_rename(self):
        self.cat_num = 0
        self.bind_num = 0
        self.pass_num = 0
        sorted_nodes = OrderedDict()
        for node in self.graph.topo_traverse():
            node: ModuleNode
            if isinstance(node.val, Module):
                pass
            elif node.is_pass_node():
                node.name = f'_{self.bind_num}'
                self.bind_num += 1
            elif node.is_cat_node():
                node.name = f'cat_{self.cat_num}'
                self.cat_num += 1
            elif node.is_bind_node():
                node.name = f'bind_{self.bind_num}'
                self.bind_num += 1
            else:
                pass
            sorted_nodes[node.name] = node
        self.graph.nodes = sorted_nodes

    # other
    def post_operations(self):
        self.add_bind_nodes()
        self.add_pass_nodes()
        self.topo_rename()


class PathToGraph(ToGraph):

    def __init__(self, path_list: PathList, model: Module) -> None:
        super().__init__()
        self.path_list = path_list
        self.cat_dict: Dict[str, str] = {}
        self.cat_num = 0
        self.bind_num = 0
        self.name2module = dict(model.named_modules())
        self.parse()

        self.post_operations()

    def parse(self):
        self.parse_unit(self.path_list, [])

    def parse_unit(self, path_unit: Union[PathList, Path, PathNode],
                   next_nodes: List[ModuleNode]):
        current_node = None
        # path_list
        if isinstance(path_unit, PathList):
            for single_path in path_unit:  # sibling
                self.parse_unit(single_path, next_nodes)

        # path:
        elif isinstance(path_unit, Path):
            current_nexts = next_nodes
            for node in path_unit:  # parent -> children
                current_node = self.parse_unit(node, current_nexts)
                current_nexts = [current_node]

        # Node
        elif isinstance(path_unit, PathNode):

            # cat node: [cat_path_lists]
            if isinstance(path_unit, PathCatNode):
                current_node = self.add_or_find_node(path_unit)
                self.connect_nexts(current_node, next_nodes)
                for catpath in path_unit.path_lists:  # sibling
                    self.parse_unit(catpath, [current_node])

            # single node
            else:
                current_node = self.add_or_find_node(path_unit)
                self.connect_nexts(current_node, next_nodes)
        return current_node

    def add_or_find_cat_node(self, pathnode: PathCatNode):

        def unify_cat_name(name: str):
            cat_name = name.split('_')
            inputs = cat_name[1:]
            inputs.sort()
            return f"cat_{'_'.join(inputs)}"

        name_id = pathnode.name
        name_id = unify_cat_name(name_id)
        if name_id in self.cat_dict:
            name = self.cat_dict[name_id]
        else:
            name = f'cat_{self.cat_num}'
            self.cat_num += 1
            self.cat_dict[name_id] = name
        node = self.graph.add_or_find_node(ModuleNode(name, 'cat'))
        return node

    def add_or_find_node(self, pathnode: PathNode) -> Module:
        if isinstance(pathnode, PathCatNode):
            return self.add_or_find_cat_node(pathnode)
        else:
            name = pathnode.name
            assert name in self.name2module, f"{name} doesn't exist in model"
            module = self.name2module[name]
            return self.graph.add_or_find_node(ModuleNode(name, module))

    def connect_nexts(self, node, nexts):
        for next in nexts:
            self.graph.connect(node, next)
