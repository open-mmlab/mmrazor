# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from mmrazor.models.mutables import DMCPChannelUnit
from mmrazor.registry import MODELS
from ...architectures import DMCPBatchNorm2d
from .channel_mutator import ChannelMutator, ChannelUnitType


@MODELS.register_module()
class DMCPChannelMutator(ChannelMutator[DMCPChannelUnit]):
    """DMCP channel mutable based channel mutator. It uses DMCPPChannelUnit.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict( type='DMCPPChannelUnit', units={}).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict( type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')).
            Change loss_calculator according to task and backbone.
        pruning_cfg (Tuple): (min_sample_rate, max_sample_rate, sample_offset)
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[ChannelUnitType]] = dict(
                     type='DMCPChannelUnit', units={}),
                 parse_cfg: Dict = dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer'),
                 pruning_cfg=(0.1, 1, 0.05),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)
        self.pruning_cfg = pruning_cfg

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Prepare from a model for pruning.

        It includes two steps:
        1. parse the model and get MutableChannelUnits.
        2. call unit.prepare_for_pruning for each unit.
        """
        super().prepare_from_supernet(supernet)
        self.prepare_arch_params(supernet)

    def _build_arch_param(self, num_choices) -> nn.Parameter:
        """Build learnable architecture parameters."""
        return nn.Parameter(torch.zeros(num_choices))

    def prepare_arch_params(self, supernet: Module) -> None:
        """Prepare the arch parameters and associate them with the
        corresponding op."""
        self.arch_params = nn.ParameterDict()
        self._op_arch_align = dict()
        self._arch_params_attr = dict()
        for group_id, module in enumerate(self.units):
            arch_message = self._generate_arch_message(
                module.mutable_channel.num_channels)
            self._arch_params_attr[str(group_id)] = arch_message
            group_arch_param = self._build_arch_param(arch_message[1])
            self.arch_params[str(group_id)] = group_arch_param

            for unit in module.output_related:
                self._op_arch_align[str(unit.name)] = str(group_id)

        self._bn_arch_align = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, DMCPBatchNorm2d):
                self._bn_arch_align[module] = self._op_arch_align[str(name)]

    def _generate_arch_message(self, out_channels: int) -> tuple:
        """
        Define the search space of the channel according to the pruning
        rate range, where the search space consists of two parts
            1. sampled by pruning rate (that is, maximum, minimum and random
                pruning rate)
            2. sampled by probability
        Inputs:
            out_channels (int): channel num of conv layers.
        Outputs:
            attr (tuple): (group_size, num_groups, min_ch)
        """
        (min_rate, max_rate, rate_offset) = self.pruning_cfg

        # sampled by probability
        group_size = int(rate_offset * out_channels / max_rate)
        num_groups = int((max_rate - min_rate) / rate_offset + 1e-4)
        min_ch = out_channels - (group_size * num_groups)
        assert min_ch > 0
        assert group_size * num_groups + min_ch == out_channels

        return (group_size, num_groups, min_ch)

    def modify_supernet_forward(self, arch_train: str) -> None:
        """According to the arch_train, assign the arch parameter to the
        forward of the corresponding op."""
        for module, group_id in self._bn_arch_align.items():
            arch_param: Optional[nn.Parameter] = None
            arch_params_attr: Optional[Tuple] = None
            if arch_train:
                arch_param = self.arch_params[self._bn_arch_align[module]]
                arch_params_attr = self._arch_params_attr[str(group_id)]
            module.set_forward_args(
                arch_param=arch_param, arch_attr=arch_params_attr)

    def sample_subnet(self, mode: str, arch_train: str) -> None:
        """Sampling according to the input mode."""
        choices = dict()

        for group_id, _ in enumerate(self.units):
            choices[str(group_id)] = self._prune_by_arch(mode, group_id)
        self.set_choices(choices)

        self.modify_supernet_forward(arch_train)

    def _prune_by_arch(self, mode: str, group_id: int) -> Union[int, Any]:
        """Prune the output channels according to the specified mode.

        Inputs:
            mode (list): one of ['max', 'min', 'random', 'direct', 'expected']
            group_id (int): index of units

        Outputs:
            channels (int): for mode 'max'/'min'/'random'/'dirext'
            channels (tensor): for mode 'expected'
        """
        arch_param = self.arch_params[str(group_id)]
        (group_size, num_groups, min_ch) =\
            self._arch_params_attr[str(group_id)]

        if mode == 'max':
            return min_ch + group_size * num_groups
        elif mode == 'min':
            return min_ch
        elif mode == 'random':
            return min_ch + group_size * random.randint(0, num_groups)
        else:
            if num_groups == 0:
                return min_ch
            prob = torch.clamp(arch_param, min=0)
            condition_prob = torch.exp(-prob)
            if mode == 'direct':
                direct_channel = min_ch
                for i in range(num_groups):
                    if random.uniform(0, 1) > condition_prob[i]:
                        break
                    direct_channel += group_size
                return direct_channel
            elif mode == 'expected':
                marginal_prob = torch.cumprod(condition_prob, dim=0)
                expected_channel = (torch.sum(marginal_prob) *
                                    group_size) + min_ch
                return expected_channel
            else:
                raise NotImplementedError

    def set_choices(self, choices: Dict[str, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.

        Args:
            choices (Dict[str, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for group_id, module in enumerate(self.units):
            if str(group_id) not in choices.keys():
                # allow optional target_prune_ratio
                continue
            choice = choices[str(group_id)]
            module.current_choice = choice
            module.mutable_channel.activated_tensor_channels = choice
