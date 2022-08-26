# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Generic, List, Optional, Set, Type, Union

from torch import Tensor
from torch.nn import Module

from mmrazor.models.architectures.dynamic_op.bricks import DynamicChannelMixin
from mmrazor.models.mutables import (MUTABLECHANNELGROUP, MutableChannelGroup,
                                     SimpleChannelGroup)
from mmrazor.registry import MODELS
from mmrazor.structures.graph import ModuleGraph
from ..base_mutator import BaseMutator


@MODELS.register_module()
class BaseChannelMutator(BaseMutator, Generic[MUTABLECHANNELGROUP]):

    # init

    def __init__(
            self,
            channl_group_cfg: Union[
                dict, Type[MutableChannelGroup]] = SimpleChannelGroup,
            # tracer_cfg=dict(type='fx'),
            tracer_cfg=dict(
                type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')),
            skip_prefixes: Optional[List[str]] = None,  # TODO: support later
            init_cfg: Optional[Dict] = None) -> None:

        super().__init__(init_cfg)

        self.tracer_cfg = tracer_cfg
        assert self.tracer_cfg['type'] in ['fx', 'BackwardTracer', 'model']

        # only record prunable group
        self._name2group: Dict[str, MUTABLECHANNELGROUP] = {}
        self.groups: Set[MUTABLECHANNELGROUP] = set()
        self.group_class, self.group_args = self._parse_group_config(
            channl_group_cfg)

    # prepare model

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Convert modules to dynamicops and parse channel groups."""

        # self.convert_dynamic_module(supernet, self.module_converters)
        supernet.eval()

        self.group_class.prepare_model(supernet)
        self._name2module = dict(supernet.named_modules())

        if self.tracer_cfg['type'] == 'BackwardTracer':
            graph = ModuleGraph.init_using_backward_tracer(
                supernet, self.tracer_cfg)
        elif self.tracer_cfg['type'] == 'fx':

            def is_dynamic_op_for_fx_tracer(module, module_name):
                """determine if a module is a dynamic op for fx tracer."""
                return isinstance(module, DynamicChannelMixin)

            graph = ModuleGraph.init_using_fx_tracer(
                supernet, is_dynamic_op_for_fx_tracer)
        else:
            raise NotImplementedError()

        self._graph = graph
        self.groups = self.group_class.parse_channel_groups(
            graph, self.group_args)
        for group in self.groups:
            group.prepare_for_pruning()
            self._name2group[group.name] = group

    # pruning structure manage

    @property
    def choice_template(self) -> Dict:
        """return the template for configurate the pruning ratio of the model.

        Example:
            {'net.3_(0, 16)_out_2_in_1': 16, 'net.0_(0, 8)_out_2_in_1': 8}
        """
        template = {}
        for group in self.prunable_groups:
            template[group.name] = group.current_choice
        return template

    def sample_choices(self) -> Dict[str, Union[int, float]]:
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].sample_choice()
        return template

    def set_choices(self, config: Dict[str, Union[int, float]]):
        for name, choice in config.items():
            group = self._name2group[name]
            group.current_choice = choice

    def fix_channel_mutables(self):
        for group in self.groups:
            group.fix_chosen()

    @property
    def current_choices(self):
        config = self.choice_template
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

    def search_groups(self) -> Dict:
        return self._name2group

    def mutable_class_type(self):
        return self.group_class

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        if name == 'model':
            object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)
