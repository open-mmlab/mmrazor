# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Type

import torch.nn as nn

from mmrazor.models.mutables import DiffMutable
from mmrazor.registry import MODELS
from .base_mutator import ArchitectureMutator


@MODELS.register_module()
class DiffMutator(ArchitectureMutator[DiffMutable]):
    """Differentiable mutable based mutator.

    `DiffMutator` is responsible for mutating `DiffMutable`, `DiffOP`,
    `DiffChoiceRoute` and `GumbelChoiceRoute`. The architecture
    parameters of the mutables are retained in `DiffMutator`.

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_group: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(custom_group=custom_group, init_cfg=init_cfg)

    def prepare_from_supernet(self, supernet: nn.Module) -> None:
        """Inherit from ``BaseMutator``'s, generate `arch_params` in DARTS.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

        super().prepare_from_supernet(supernet)
        self.arch_params = self.build_arch_params()

    def build_arch_params(self):
        """This function will build many arch params, which are generally used
        in differentiable search algorithms, such as Darts' series. Each
        group_id corresponds to an arch param, so the Mutables with the same
        group_id share the same arch param.

        Returns:
            torch.nn.ParameterDict: the arch params are got by `search_group`.
        """

        arch_params: Dict[int, nn.Parameter] = dict()

        for group_id, modules in self.search_group.items():
            group_arch_param = modules[0].build_arch_param()
            arch_params[group_id] = group_arch_param

        return arch_params

    def modify_supernet_forward(self):
        """Modify the DiffMutable's default arch_param in forward.

        In MMRazor, the `arch_param` is along to `DiffMutator`, while the
        `DiffMutable` needs `arch_param` in the forward. Here we use partial
        function to assign the corresponding `arch_param` to each
        `DiffMutable`.
        """

        for group_id, modules in self.search_group.items():
            for module in modules:
                module.set_forward_args(arch_param=self.arch_params[group_id])

    @property
    def mutable_class_type(self) -> Type[DiffMutable]:
        """Differentiable mutable class type.

        Returns:
            Type[DiffMutable]: Class type of differentiable mutable.
        """
        return DiffMutable
