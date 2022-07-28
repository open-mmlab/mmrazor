# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Type

from torch.nn import Module

from mmrazor.models.mutables import OneShotMutableChannel
from mmrazor.registry import MODELS
from ..base_mutator import BaseMutator
from ..mixins import DynamicSampleMixin, GroupMixin


# TODO
# this is just for demo and will be deleted after channel mutator is done.
@MODELS.register_module()
class BigNASChannelMutator(BaseMutator[OneShotMutableChannel], GroupMixin,
                           DynamicSampleMixin):
    """The base class for mutable based mutator.

    All subclass should implement the following APIS:

    - ``mutable_class_type``

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_group: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        if custom_group is None:
            custom_group = []
        self._custom_group = custom_group
        self._search_groups: Optional[Dict[int,
                                           List[OneShotMutableChannel]]] = None

    # TODO
    # should be a class property
    @property
    def mutable_class_type(self) -> Type[OneShotMutableChannel]:
        return OneShotMutableChannel

    def prepare_from_supernet(self, supernet: Module) -> None:
        self._build_search_groups(supernet)

    @property
    def search_groups(self) -> Dict[int, List[OneShotMutableChannel]]:
        if self._search_groups is None:
            raise RuntimeError(
                'Call `prepare_from_supernet` before access search group!')
        return self._search_groups

    def _build_search_groups(self, supernet: Module) -> None:
        self._search_groups = self.group_by_name_and_alias(
            supernet, self._custom_group)

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
