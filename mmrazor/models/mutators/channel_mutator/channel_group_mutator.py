# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Generic, List, Optional, Set, Type, Union

from torch.nn import Module

from ....registry import MODELS, TASK_UTILS
from ....structures import BackwardTracer
from ....structures.graph import ModuleGraph
from ...architectures.dynamic_op.base import ChannelDynamicOP
from ...mutables.mutable_channel.channel_groups.mutable_channel_group import (
    MUTABLECHANNELGROUP, MutableChannelGroup)
from ...mutables.mutable_channel.channel_groups.Simple_channel_group import \
    SimpleChannelGroup
from ...mutables.mutable_channel.simple_mutable_channel import \
    SimpleMutableChannel
from .channel_mutator import ChannelMutator


@MODELS.register_module()
class ChannelGroupMutator(ChannelMutator, Generic[MUTABLECHANNELGROUP]):

    # init

    def __init__(
            self,
            model: Module,
            channl_group_cfg: Union[
                dict, Type[MUTABLECHANNELGROUP]] = SimpleChannelGroup,
            # tracer_cfg=dict(type='fx'),
            tracer_cfg=dict(
                type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')),
            skip_prefixes: Optional[List[str]] = None,
            init_cfg: Optional[Dict] = None) -> None:

        mutable_cfg = {'type': 'MutableMask'}
        super().__init__(mutable_cfg, None, skip_prefixes, init_cfg)

        self.model = model
        self.tracer_cfg = tracer_cfg
        assert self.tracer_cfg['type'] in ['fx', 'BackwardTracer', 'model']

        # only record prunable group
        self._name2group: Dict[str, MUTABLECHANNELGROUP] = {}
        self.groups: Set[MUTABLECHANNELGROUP] = set()
        self.group_class, self.group_args = self._parse_group_config(
            channl_group_cfg)

        self.prepare_from_supernet(self.model)

    # prepare model

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Convert modules to dynamicops and parse channel groups."""

        # self.convert_dynamic_module(supernet, self.module_converters)
        supernet.eval()

        def is_dynamic_op(module, module_name):
            """determine if a module is a dynamic op for fx tracer."""
            return isinstance(module, ChannelDynamicOP)

        self.group_class.prepare_model(supernet)
        self._name2module = dict(supernet.named_modules())

        if self.tracer_cfg['type'] == 'BackwardTracer':
            self.tracer: BackwardTracer = TASK_UTILS.build(self.tracer_cfg)
            graph = ModuleGraph.init_using_backward_tracer(
                supernet, self.tracer)
        elif self.tracer_cfg['type'] == 'fx':
            graph = ModuleGraph.init_using_fx_tracer(supernet, is_dynamic_op)
        else:
            raise NotImplementedError()

        self._graph = graph
        self.groups = self.group_class.parse_channel_groups(
            graph, self.group_args)
        for group in self.groups:
            group.prepare_for_pruning()
            self._name2group[group.name] = group

    # pruning structure manage

    def structure_template(self) -> Dict:
        """return the template for configurate the pruning ratio of the model.

        Example:
            {'net.3_(0, 16)_out_2_in_1': 16, 'net.0_(0, 8)_out_2_in_1': 8}
        """
        templabe = {}
        for group in self.prunable_groups:
            templabe[group.name] = group.current_choice
        return templabe

    def sample_structure(self) -> Dict[str, Union[int, float]]:
        template = self.structure_template()
        for key in template:
            template[key] = self._name2group[key].sample_choice()
        return template

    def apply_structure(self, config: Dict[str, Union[int, float]]):
        for name, choice in config.items():
            group = self._name2group[name]
            group.current_choice = choice

    @property
    def current_structure(self):
        config = self.structure_template()
        for group in self.prunable_groups:
            config[group.name] = group.current_choice
        return config

    # group manage

    @property
    def prunable_groups(self) -> List[MUTABLECHANNELGROUP]:
        return [group for group in self.groups if group.is_prunable]

    def _parse_group_config(self, group_cfg):
        if isinstance(group_cfg, dict):
            group_class = MODELS.module_dict[group_cfg['type']]
            group_args = copy.copy(group_cfg)
            group_args.pop('type')
        elif issubclass(group_cfg, MutableChannelGroup):
            group_class = group_cfg
            group_args = {}
        else:
            raise NotImplementedError()
        return group_class, group_args

    # implementation of abstract functions

    def build_search_groups(self, supernet: Module):
        """GROUP is used to replace search_groups, but it's more complicated to
        handle different pruning algorithms."""
        return self.prunable_groups

    def mutable_class_type(self):
        return SimpleMutableChannel
