# Copyright (c) OpenMMLab. All rights reserved.
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

        >>> # Assume that a toy model consists of three mutables
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
    """

    @property
    def mutable_class_type(self):
        """One-shot mutable class type.

        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return OneShotMutableModule
