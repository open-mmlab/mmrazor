# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, Dict, List, Optional

from torch.nn import Module

from mmrazor.registry import MODELS
from ...mutables import OneShotMutableChannel
from .channel_mutator import ChannelMutator


@MODELS.register_module()
class OneShotChannelMutator(ChannelMutator):
    """One-shot channel mutable based channel mutator.

    Args:
        mutable_cfg (dict): The config for the channel mutable.
        tracer_cfg (dict | Optional): The config for the model tracer.
            We Trace the topology of a given model with the tracer.
        skip_prefixes (List[str] | Optional): The module whose name start with
            a string in skip_prefixes will not be pruned.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 mutable_cfg: Dict,
                 tracer_cfg: Optional[Dict] = None,
                 skip_prefixes: Optional[List[str]] = None,
                 force_link: Optional[bool] = True,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(mutable_cfg, tracer_cfg, skip_prefixes, init_cfg)
        # TODO(shiguang): make mutable_cfg optional?
        self.force_link = force_link

    def sample_choices(self):
        """Sample a choice that records a selection from the search space.

        Returns:
            dict: Record the information to build the subnet from the supernet.
                Its keys are the properties ``group_idx`` in the channel
                mutator's ``search_groups``, and its values are the sampled
                choice.
        """
        choice_dict = dict()
        for group_idx, mutables in self.search_groups.items():
            choice_dict[group_idx] = mutables[0].sample_choice()
        return choice_dict

    def set_choices(self, choice_dict: Dict[int, Any]) -> None:
        """Set current subnet according to ``choice_dict``.

        Args:
            choice_dict (Dict[int, Any]): Choice dict.
        """
        for group_idx, choice in choice_dict.items():
            mutables = self.search_groups[group_idx]
            for mutable in mutables:
                mutable.current_choice = choice

    def set_max_choices(self) -> None:
        """Set the channel numbers of each layer to maximum."""
        for mutables in self.search_groups.values():
            for mutable in mutables:
                mutable.current_choice = mutable.max_choice

    def set_min_choices(self) -> None:
        """Set the channel numbers of each layer to minimum."""
        for mutables in self.search_groups.values():
            for mutable in mutables:
                mutable.current_choice = mutable.min_choice

    # todo: check search gorups
    def build_search_groups(self, supernet: Module):
        """Build `search_groups`. The mutables in the same group should be
        pruned together.

        Examples:
            >>> class ResBlock(nn.Module):
            ...     def __init__(self) -> None:
            ...         super().__init__()
            ...
            ...         self.op1 = nn.Conv2d(3, 8, 1)
            ...         self.bn1 = nn.BatchNorm2d(8)
            ...         self.op2 = nn.Conv2d(8, 8, 1)
            ...         self.bn2 = nn.BatchNorm2d(8)
            ...         self.op3 = nn.Conv2d(8, 8, 1)
            ...
            ...      def forward(self, x):
            ...         x1 = self.bn1(self.op1(x))
            ...         x2 = self.bn2(self.op2(x1))
            ...         x3 = self.op3(x2 + x1)
            ...         return x3

            >>> class ToyPseudoLoss:
            ...
            ...     def __call__(self, model):
            ...         pseudo_img = torch.rand(2, 3, 16, 16)
            ...         pseudo_output = model(pseudo_img)
            ...         return pseudo_output.sum()

            >>> mutator = OneShotChannelMutator(
            ...     tracer_cfg=dict(type='BackwardTracer',
            ...         loss_calculator=ToyPseudoLoss()),
            ...     mutable_cfg=dict(type='OneShotMutableChannel',
            ...         candidate_choices=[4 / 8, 1.0], candidate_mode='ratio')

            >>> model = ResBlock()
            >>> mutator.prepare_from_supernet(model)
            >>> mutator.search_groups
            {0: [OneShotMutableChannel(name=op2, ...), # mutable out
                 OneShotMutableChannel(name=op1, ...), # mutable out
                 OneShotMutableChannel(name=op3, ...), # mutable in
                 OneShotMutableChannel(name=op2, ...), # mutable in
                 OneShotMutableChannel(name=bn2, ...), # mutable out
                 OneShotMutableChannel(name=bn1, ...)] # mutable out
            }
        """
        groups = self.find_same_mutables(supernet, self.force_link)

        search_groups = dict()
        group_idx = 0
        for group in groups.values():
            is_skip = False
            for mutable in group:
                if self.is_skip_pruning(mutable.name, self.skip_prefixes):
                    warnings.warn(f'Group {group} is not searchable due to'
                                  f' skip_prefixes: {self.skip_prefixes}')
                    is_skip = True
                    break
            if not is_skip:
                search_groups[group_idx] = group
                group_idx += 1

        return search_groups

    def mutable_class_type(self):
        """One-shot channel mutable class type.

        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return OneShotMutableChannel
