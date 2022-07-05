# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

from mmrazor.registry import MODELS
from ...mutables import OneShotMutableModule
from .module_mutator import ModuleMutator


@MODELS.register_module()
class OneShotModuleMutator(ModuleMutator):
    """One-shot mutable based mutator.

    Examples:
        >>> mutator = OneShotModuleMutator()
        >>> mutator.mutable_class_type
        <class 'mmrazor.models.mutables.oneshot_mutable.OneShotMutable'>

        >>> # Assume that a toy model consists of three mutabels
        >>> # whose name are op1,op2,op3.
        >>> # Each mutable contains 4 choices: choice1, choice2,
        >>> # choice3 and choice4.
        >>> supernet = ToyModel()
        >>> name2module = dict(supernet.named_modules())
        >>> [name for name, module in name2module.items() if isinstance(module, mutator.mutable_class_type)]  # noqa E501
        ['op1', 'op2', 'op3']

        >>> mutator.prepare_from_supernet(supernet)
        >>> mutator.search_groups
        {0: [op1], 1: [op2], 2: [op3]}

        >>> random_choices = mutator.sample_choices()
        {0: 'choice1', 1: 'choice2', 2: 'choice3'}
        >>> mutator.set_subnet(random_choices)

        >>> supernet.op1.current_choice
        'choice1'
        >>> supernet.op2.current_choice
        'choice2'
        >>> supernet.op3.current_choice
        'choice3'

        >>> random_choices_ = mutator.sample_choices()
        {0: 'choice3', 1: 'choice2', 2: 'choice1'}
        >>> mutator.set_subnet(random_choices_)

        >>> supernet.op1.current_choice
        'choice3'
        >>> supernet.op2.current_choice
        'choice2'
        >>> supernet.op3.current_choice
        'choice1'
    """

    def sample_choices(self) -> Dict[int, Any]:
        """Sampling by search groups.

        The sampling result of the first mutable of each group is the sampling
        result of this group.

        Returns:
            Dict[int, Any]: Random choices dict.
        """
        random_choices = dict()
        for group_id, modules in self.search_groups.items():
            random_choices[group_id] = modules[0].sample_choice()

        return random_choices

    def set_choices(self, choices: Dict[int, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.

        Args:
            choices (Dict[int, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for group_id, modules in self.search_groups.items():
            choice = choices[group_id]
            for module in modules:
                module.current_choice = choice

    @property
    def mutable_class_type(self):
        """One-shot mutable class type.

        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return OneShotMutableModule
