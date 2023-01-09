# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from mmrazor.registry import MODELS
from ...mutables import DiffMutableModule
from .module_mutator import ModuleMutator


@MODELS.register_module()
class DiffModuleMutator(ModuleMutator):
    """Differentiable mutable based mutator.

    `DiffModuleMutator` is responsible for mutating `DiffMutableModule`,
    `DiffMutableOP`, `DiffChoiceRoute` and `GumbelChoiceRoute`.
    The architecture parameters of the mutables are retained
    in `DiffModuleMutator`.

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_groups: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(custom_groups=custom_groups, init_cfg=init_cfg)

    @property
    def mutable_class_type(self):
        """Differentiable mutable class type.

        Returns:
            Type[DiffMutableModule]: Class type of differentiable mutable.
        """
        return DiffMutableModule
