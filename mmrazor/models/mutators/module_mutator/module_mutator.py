# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Type

from torch.nn import Module

from ..base_mutator import MUTABLE_TYPE, BaseMutator


class ModuleMutator(BaseMutator[MUTABLE_TYPE]):
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
        self._build_search_groups(supernet)

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

    def _build_name_mutable_mapping(
            self, supernet: Module) -> Dict[str, MUTABLE_TYPE]:
        """Mapping module name to mutable."""
        name2mutable: Dict[str, MUTABLE_TYPE] = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                name2mutable[name] = module
        self._name2mutable = name2mutable

        return name2mutable

    def _build_alias_names_mapping(self,
                                   supernet: Module) -> Dict[str, List[str]]:
        """Mapping alias to module names."""
        alias2mutable_names: Dict[str, List[str]] = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                if module.alias is not None:
                    if module.alias not in alias2mutable_names:
                        alias2mutable_names[module.alias] = [name]
                    else:
                        alias2mutable_names[module.alias].append(name)

        return alias2mutable_names

    def _build_search_groups(self, supernet: Module) -> None:
        """Build search group with ``custom_group`` and ``alias``(see more
        information in :class:`BaseMutable`). Grouping by alias and module name
        are both supported.

        Note:
            Apart from user-defined search group, all other searchable
            modules(mutable) will be grouped separately.

            The main difference between using alias and module name for
            grouping is that the alias is One-to-Many while the module
            name is One-to-One.

            When using both alias and module name in `custom_group`, the
            priority of alias is higher than that of module name.

            If alias is set in `custom_group`, then its corresponding module
            name should not be in the `custom_group`.

            Moreover, there should be no duplicate keys in the `custom_group`.

        Example:
            >>> import torch
            >>> from mmrazor.models.mutables.diff_mutable import DiffMutableOP

            >>> # Assume that a toy model consists of three mutables
            >>> # whose name are op1,op2,op3. The corresponding
            >>> # alias names of the three mutables are a1, a1, a2.
            >>> model = ToyModel()

            >>> # Using alias for grouping
            >>> mutator = DiffMutableOP(custom_group=[['a1'], ['a2']])
            >>> mutator.prepare_from_supernet(model)
            >>> mutator.search_groups
            {0: [op1, op2], 1: [op3]}

            >>> # Using module name for grouping
            >>> mutator = DiffMutableOP(custom_group=[['op1', 'op2'], ['op3']])
            >>> mutator.prepare_from_supernet(model)
            >>> mutator.search_groups
            {0: [op1, op2], 1: [op3]}

            >>> # Using both alias and module name for grouping
            >>> mutator = DiffMutableOP(custom_group=[['a2'], ['op2']])
            >>> mutator.prepare_from_supernet(model)
            >>> # The last operation would be grouped
            >>> mutator.search_groups
            {0: [op3], 1: [op2], 2: [op1]}


        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        name2mutable = self._build_name_mutable_mapping(supernet)
        alias2mutable_names = self._build_alias_names_mapping(supernet)

        # Check whether the custom group is valid
        if len(self._custom_group) > 0:
            self._check_valid_groups(alias2mutable_names, name2mutable,
                                     self._custom_group)

        # Construct search_groups based on user-defined group
        search_groups: Dict[int, List[MUTABLE_TYPE]] = dict()

        current_group_nums = 0
        grouped_mutable_names: List[str] = list()
        grouped_alias: List[str] = list()
        for group in self._custom_group:
            group_mutables = list()
            for item in group:
                if item in alias2mutable_names:
                    # if the item is from alias name
                    mutable_names: List[str] = alias2mutable_names[item]
                    grouped_alias.append(item)
                    group_mutables.extend(
                        [name2mutable[n] for n in mutable_names])
                    grouped_mutable_names.extend(mutable_names)
                else:
                    # if the item is in name2mutable
                    group_mutables.append(name2mutable[item])
                    grouped_mutable_names.append(item)

            search_groups[current_group_nums] = group_mutables
            current_group_nums += 1

        # Construct search_groups based on alias
        for alias, mutable_names in alias2mutable_names.items():
            if alias not in grouped_alias:
                # Check whether all current names are already grouped
                flag_all_grouped = True
                for mutable_name in mutable_names:
                    if mutable_name not in grouped_mutable_names:
                        flag_all_grouped = False

                # If not all mutables are already grouped
                if not flag_all_grouped:
                    search_groups[current_group_nums] = []
                    for mutable_name in mutable_names:
                        if mutable_name not in grouped_mutable_names:
                            search_groups[current_group_nums].append(
                                name2mutable[mutable_name])
                            grouped_mutable_names.append(mutable_name)
                    current_group_nums += 1

        # check whether all the mutable objects are in the search_groups
        for name, module in supernet.named_modules():
            if isinstance(module, self.mutable_class_type):
                if name in grouped_mutable_names:
                    continue
                else:
                    search_groups[current_group_nums] = [module]
                    current_group_nums += 1

        grouped_counter = Counter(grouped_mutable_names)

        # find duplicate keys
        duplicate_keys = list()
        for key, count in grouped_counter.items():
            if count > 1:
                duplicate_keys.append(key)

        assert len(grouped_mutable_names) == len(
            list(set(grouped_mutable_names))), \
            'There are duplicate keys in grouped mutable names. ' \
            f'The duplicate keys are {duplicate_keys}. ' \
            'Please check if there are duplicate keys in the `custom_group`.'

        self._search_groups = search_groups

    def _check_valid_groups(self, alias2mutable_names: Dict[str, List[str]],
                            name2mutable: Dict[str, MUTABLE_TYPE],
                            custom_group: List[List[str]]) -> None:

        aliases = [*alias2mutable_names.keys()]
        module_names = [*name2mutable.keys()]

        # check if all keys are legal
        expanded_custom_group: List[str] = [
            _ for group in custom_group for _ in group
        ]
        legal_keys: List[str] = [*aliases, *module_names]

        for key in expanded_custom_group:
            if key not in legal_keys:
                raise AssertionError(
                    f'The key: {key} in `custom_group` is not legal. '
                    f'Legal keys are: {legal_keys}. '
                    'Make sure that the keys are either alias or mutable name')

        # when the mutable has alias attribute, the corresponding module
        # name should not be used in `custom_group`.
        used_aliases = list()
        for group in custom_group:
            for key in group:
                if key in aliases:
                    used_aliases.append(key)

        for alias_key in used_aliases:
            mutable_names: List = alias2mutable_names[alias_key]
            # check whether module name is in custom group
            for mutable_name in mutable_names:
                if mutable_name in expanded_custom_group:
                    raise AssertionError(
                        f'When a mutable is set alias attribute :{alias_key},'
                        f'the corresponding module name {mutable_name} should '
                        f'not be used in `custom_group` {custom_group}.')
