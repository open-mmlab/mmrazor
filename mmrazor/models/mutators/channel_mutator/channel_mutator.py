# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import abstractmethod
from typing import Callable, Dict, List, Optional

import torch.nn as nn
from torch.nn import Module
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import (BatchNorm1d, BatchNorm2d, BatchNorm3d,
                                        _BatchNorm)
from torch.nn.modules.instancenorm import (InstanceNorm1d, InstanceNorm2d,
                                           InstanceNorm3d, _InstanceNorm)

from mmrazor.core.tracer import (ConcatNode, ConvNode, DepthWiseConvNode,
                                 LinearNode, NormNode, PathList)
from mmrazor.models.architectures.dynamic_op import (build_dynamic_bn,
                                                     build_dynamic_conv2d,
                                                     build_dynamic_gn,
                                                     build_dynamic_in,
                                                     build_dynamic_linear)
from mmrazor.models.mutables import OneShotChannelMutable
from mmrazor.registry import MODELS, TASK_UTILS
from ..base_mutator import BaseMutator

NONPASS_NODES = (ConvNode, LinearNode, ConcatNode)
PASS_NODES = (NormNode, DepthWiseConvNode)

NONPASS_MODULES = (nn.Conv2d, nn.Linear)
PASS_MODULES = (_BatchNorm, _InstanceNorm, GroupNorm)


@MODELS.register_module()
class ChannelMutator(BaseMutator):
    """Base class for channel-based mutators.

    Args:
        mutable_cfg (dict): The config for the channel mutable.
        tracer_cfg (dict | Optional): The config for the model tracer.
            We Trace the topology of a given model with the tracer.
        skip_prefixes (List[str] | Optional): The module whose name start with
            a string in skip_prefixes will not be pruned.
        init_cfg (dict, optional): The config to control the initialization.

    Attributes:
        search_groups (Dict[int, List]): Search group of supernet. Note that
            the search group of a mutable based channel mutator is composed of
            corresponding mutables. Mutables in the same search group should
            be pruned together.
        name2module (Dict[str, :obj:`torch.nn.Module`]): The mapping from
            a module name to the module.

    Notes:
        # To avoid ambiguity, we only allow the following two cases:
        # 1. None of the parent nodes of a node is a `ConcatNode`
        # 2. A node has only one parent node which is a `ConcatNode`
    """

    def __init__(
        self,
        mutable_cfg: Dict,
        tracer_cfg: Optional[Dict] = None,
        skip_prefixes: Optional[List[str]] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        self.mutable_cfg = mutable_cfg
        if tracer_cfg:
            self.tracer = TASK_UTILS.build(tracer_cfg)
        else:
            self.tracer = None
        self.skip_prefixes = skip_prefixes
        self._search_groups: Optional[Dict[int, List[Module]]] = None

    def add_link(self, path_list: PathList) -> None:
        """Establish the relationship between the current nodes and their
        parents."""
        for path in path_list:
            pre_node = None
            for node in path:
                if isinstance(node, DepthWiseConvNode):
                    module = self.name2module[node.name]
                    # The in_channels and out_channels of a depth-wise conv
                    # should be the same
                    module.mutable_out.register_same_mutable(module.mutable_in)
                    module.mutable_in.register_same_mutable(module.mutable_out)

                if isinstance(node, ConcatNode):
                    if pre_node is not None:
                        module_names = node.get_module_names()
                        concat_modules = [
                            self.name2module[name] for name in module_names
                        ]
                        concat_mutables = [
                            module.mutable_out for module in concat_modules
                        ]
                        pre_module = self.name2module[pre_node.name]
                        pre_module.mutable_in.register_same_mutable(
                            concat_mutables)

                    for cur_path_list in node:
                        self.add_link(cur_path_list)

                    # ConcatNode is the last node in a path
                    break

                if pre_node is None:
                    pre_node = node
                    continue

                pre_module = self.name2module[pre_node.name]
                cur_module = self.name2module[node.name]
                pre_module.mutable_in.register_same_mutable(
                    cur_module.mutable_out)
                cur_module.mutable_out.register_same_mutable(
                    pre_module.mutable_in)

                pre_node = node

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        We support the following two cases:

        Case 1: The input is the original nn.Module. We first replace the
        conv/linear/norm modules in the input supernet with dynamic ops.
        And trace the topology of the supernet. Finally, `search_groups` can be
        built based on the topology.

        Case 2: The input supernet is made up of dynamic ops. In this case,
        relationship between nodes and their parents must have been
        established and topology of the supernet is available for us. Then
        `search_groups` can be built based on the topology.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        if self.tracer is not None:
            self.convert_dynamic_module(supernet, self.dynamic_layer)
            # The mapping from a module name to the module
            self._name2module = dict(supernet.named_modules())

            assert self.tracer is not None
            module_path_list: PathList = self.tracer.trace(supernet)

            self.add_link(module_path_list)
        else:
            self._name2module = dict(supernet.named_modules())

        self._search_groups = self.build_search_groups(supernet)

    @staticmethod
    def find_same_mutables(supernet) -> Dict:
        """The mutables in the same group should be pruned together."""
        visited = []
        groups = {}
        group_idx = 0
        for name, module in supernet.named_modules():
            if isinstance(module, OneShotChannelMutable):
                same_mutables = module.same_mutables
                if module not in visited and len(same_mutables) > 0:
                    groups[group_idx] = [module] + same_mutables
                    visited.extend(groups[group_idx])
                    group_idx += 1
        return groups

    def convert_dynamic_module(self, supernet: Module, dynamic_layer: Dict):
        """Replace the conv/linear/norm modules in the input supernet with
        dynamic ops.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be converted
                in your algorithm.
            dynamic_layer (Dict): The mapping from the module type to the
                corresponding dynamic layer.
        """

        def traverse(module, prefix):
            for name, child in module.named_children():
                module_name = prefix + name
                if isinstance(child, NONPASS_MODULES):
                    mutable_cfg = copy.deepcopy(self.mutable_cfg)
                    # mutable_cfg.update(dict(name=module_name))
                    layer = dynamic_layer[type(child)](child, module_name,
                                                       mutable_cfg,
                                                       mutable_cfg)
                    setattr(module, name, layer)
                elif isinstance(child, PASS_MODULES):
                    mutable_cfg = copy.deepcopy(self.mutable_cfg)
                    # mutable_cfg.update(dict(name=module_name))
                    layer = dynamic_layer[type(child)](child, module_name,
                                                       mutable_cfg)
                    setattr(module, name, layer)
                else:
                    traverse(child, module_name + '.')

        traverse(supernet, '')

    @abstractmethod
    def build_search_groups(self, supernet: Module):
        """Build `search_groups`.

        The mutables in the same group should be pruned together.
        """

    @property
    def search_groups(self) -> Dict[int, List]:
        """Search group of supernet.

        Note:
            For mutable based mutator, the search group is composed of
            corresponding mutables.

        Raises:
            RuntimeError: Called before search group has been built.

        Returns:
            Dict[int, List[MUTABLE_TYPE]]: Search group.
        """
        if self._search_groups is None:
            raise RuntimeError(
                'Call `search_groups` before access `build_search_groups`!')
        return self._search_groups

    @property
    def name2module(self):
        """The mapping from a module name to the module.

        Returns:
            dict: The name to module mapping.
        """
        if hasattr(self, '_name2module'):
            return self._name2module
        else:
            raise RuntimeError('Called before access `prepare_from_supernet`!')

    @property
    def dynamic_layer(self) -> Dict:
        """The mapping from a type to the corresponding dynamic layer. It is
        called in `prepare_from_supernet`.

        Returns:
            dict: The mapping dict.
        """

        dynamic_layer: Dict[Callable, Callable] = {
            nn.Conv2d: build_dynamic_conv2d,
            nn.Linear: build_dynamic_linear,
            BatchNorm1d: build_dynamic_bn,
            BatchNorm2d: build_dynamic_bn,
            BatchNorm3d: build_dynamic_bn,
            InstanceNorm1d: build_dynamic_in,
            InstanceNorm2d: build_dynamic_in,
            InstanceNorm3d: build_dynamic_in,
            GroupNorm: build_dynamic_gn
        }

        return dynamic_layer

    def is_skip_pruning(self, module_name: str,
                        skip_prefixes: Optional[List[str]]) -> bool:
        """Judge if the module with the input `module_name` should not be
        pruned.

        Args:
            module_name (str): Module name.
            skip_prefixes (list or None): The module whose name start with
                a string in skip_prefixes will not be prune.
        """
        skip_pruning = False
        if skip_prefixes:
            for prefix in skip_prefixes:
                if module_name.startswith(prefix):
                    skip_pruning = True
                    break
        return skip_pruning
