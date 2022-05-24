# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Type

from mmrazor.models.mutables import OneShotMutable
from mmrazor.registry import MODELS
from .base_mutator import ArchitectureMutator


@MODELS.register_module()
class OneShotMutator(ArchitectureMutator[OneShotMutable]):
    """One-shot mutable based mutator.

    Examples:
        >>> class SearchableModel(nn.Module):
        >>>     def __init__(self, one_shot_op_cfg):
        >>>         # assume `OneShotOP` contains 4 choices:
        >>>         # choice1, choice2, choice3 and choice4
        >>>         self.op1 = OneShotOP(**one_shot_op_cfg)
        >>>         self.op2 = OneShotOP(**one_shot_op_cfg)
        >>>         self.op3 = OneShotOP(**one_shot_op_cfg)

        >>> supernet = SearchableModel(one_shot_op_cfg)
        >>> [name for name, _ in supernet.named_children()]
        ['op1', 'op2', 'op3']
        >>> mutator1 = OneShotMutator()
        >>> mutator1.mutable_class_type
        <class 'mmrazor.models.mutables.oneshot_mutable.OneShotMutable'>
        >>> mutator1.prepare_from_supernet(supernet)
        >>> mutator1.search_group.keys()
        dict_keys([0, 1, 2])
        >>> mutator1.random_subnet
        {0: 'choice4', 1: 'choice3', 2: 'choice2'}
        >>> mutator1.set_subnet(mutator1.random_subnet)

        >>> custom_group = [
        >>>     ['op1', 'op2'],
        >>>     ['op3']
        >>> ]
        >>> mutator2 = OneShotMutator(custom_group)
        >>> mutator2.prepare_from_supernet(supernet)
        >>> mutator2.search_group.keys()
        dict_keys([0, 1])
        >>> mutator2.random_subnet
        {0: 'choice1', 1: 'choice1'}
        >>> mutator2.set_subnet(mutator2.random_subnet)
    """

    @property
    def random_subnet(self) -> Dict[int, Any]:
        """A subnet dict that records an arbitrary selection from the search
        space.

        Returns:
            Dict[int, Any]: Random subnet dict.
        """
        random_subnet = dict()
        for group_id, modules in self.search_group.items():
            random_subnet[group_id] = modules[0].random_choice

        return random_subnet

    def set_subnet(self, subnet_dict: Dict[int, Any]) -> None:
        """Set current subnet according to ``subnet_dict``.

        Args:
            subnet_dict (Dict[int, Any]): Subnet dict.
        """
        for group_id, modules in self.search_group.items():
            choice = subnet_dict[group_id]
            for module in modules:
                module.set_forward_args(choice)

    @property
    def mutable_class_type(self) -> Type[OneShotMutable]:
        """One-shot mutable class type.

        Returns:
            Type[OneShotMutable]: Class type of one-shot mutable.
        """
        return OneShotMutable
