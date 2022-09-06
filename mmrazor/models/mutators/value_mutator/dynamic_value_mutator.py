# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.registry import MODELS
from .value_mutator import ValueMutator


@MODELS.register_module()
class DynamicValueMutator(ValueMutator):

    @property
    def mutable_class_type(self):
        return OneShotMutableValue

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
