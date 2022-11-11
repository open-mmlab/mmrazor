# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Type

from torch.nn import Module

from ..base_mutator import MUTABLE_TYPE, BaseMutator
from ..group_mixin import GroupMixin


class ModuleMutator(BaseMutator[MUTABLE_TYPE], GroupMixin):
    """The base class for mutable based mutator.

    All subclass should implement the following APIS:

    - ``mutable_class_type``

    Args:
        custom_groups (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_groups: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        if custom_groups is None:
            custom_groups = []
        self._custom_groups = custom_groups
        self._search_groups: Optional[Dict[int, List[MUTABLE_TYPE]]] = None

    # TODO
    # should be a class property
    @property
    @abstractmethod
    def mutable_class_type(self) -> Type[MUTABLE_TYPE]:
        """Corresponding mutable class type.

        Returns:
            Type[MUTABLE_TYPE]: Mutable class type.
        """

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Note:
            For mutable based mutator, we need to build search group first.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        self._search_groups = self.build_search_groups(supernet,
                                                       self.mutable_class_type,
                                                       self._custom_groups)

    @property
    def name2mutable(self) -> Dict[str, MUTABLE_TYPE]:
        """Search space of supernet.

        Note:
            To get the mapping: module name to mutable.

        Raises:
            RuntimeError: Called before search space has been parsed.

        Returns:
            Dict[str, MUTABLE_TYPE]: The name2mutable dict.
        """
        if self._name2mutable is None:
            raise RuntimeError(
                'Call `prepare_from_supernet` before access name2mutable!')
        return self._name2mutable

    @property
    def search_groups(self) -> Dict[int, List[MUTABLE_TYPE]]:
        """Search group of supernet.

        Note:
            For mutable based mutator, the search group is composed of
            corresponding mutables.

        Raises:
            RuntimeError: Called before search group has been built.

        Returns:
            Dict[int, List[MUTABLE_TYPE]]: Search group.
        """
        if self._search_groups is None:
            raise RuntimeError(
                'Call `prepare_from_supernet` before access search group!')
        return self._search_groups
