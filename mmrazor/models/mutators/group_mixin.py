# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections import Counter
from typing import Dict, List, Type

from torch.nn import Module

from ..mutables import BaseMutable

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol


class GroupMixin():
    """A mixin for :class:`BaseMutator`, which can group mutables by
    ``custom_group`` and ``alias``(see more information in
    :class:`BaseMutable`). Grouping by alias and module name are both
    supported.

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
        >>> from mmrazor.models import DiffModuleMutator

        >>> # Assume that a toy model consists of three mutables
        >>> # whose name are op1,op2,op3. The corresponding
        >>> # alias names of the three mutables are a1, a1, a2.
        >>> model = ToyModel()

        >>> # Using alias for grouping
        >>> mutator = DiffModuleMutator(custom_group=[['a1'], ['a2']])
        >>> mutator.prepare_from_supernet(model)
        >>> mutator.search_groups
        {0: [op1, op2], 1: [op3]}

        >>> # Using module name for grouping
        >>> mutator = DiffModuleMutator(custom_group=[['op1', 'op2'], ['op3']])

        >>> # Using module name for grouping
        >>> mutator.prepare_from_supernet(model)
        >>> mutator.search_groups
        {0: [op1, op2], 1: [op3]}

        >>> # Using both alias and module name for grouping
        >>> mutator = DiffModuleMutator(custom_group=[['a2'], ['op2']])
        >>> mutator.prepare_from_supernet(model)
        >>> # The last operation would be grouped
        >>> mutator.search_groups
        {0: [op3], 1: [op2], 2: [op1]}

    """

    def _build_name_mutable_mapping(
            self, supernet: Module,
            support_mutables: Type) -> Dict[str, BaseMutable]:
        """Mapping module name to mutable."""
        name2mutable: Dict[str, BaseMutable] = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, support_mutables):
                name2mutable[name] = module
        self._name2mutable = name2mutable

        return name2mutable

    def _build_alias_names_mapping(
            self, supernet: Module,
            support_mutables: Type) -> Dict[str, List[str]]:
        """Mapping alias to module names."""
        alias2mutable_names: Dict[str, List[str]] = dict()
        for name, module in supernet.named_modules():
            if isinstance(module, support_mutables):

                if module.alias is not None:
                    if module.alias not in alias2mutable_names:
                        alias2mutable_names[module.alias] = [name]
                    else:
                        alias2mutable_names[module.alias].append(name)

        return alias2mutable_names

    def build_search_groups(self, supernet: Module, support_mutables: Type,
                            custom_groups: List[List[str]]) -> Dict[int, List]:
        """Build search group with ``custom_group`` and ``alias``(see more
        information in :class:`BaseMutable`). Grouping by alias and module name
        are both supported.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
            support_mutables (Type): Mutable type that can be grouped.
            custom_group (list, optional): User-defined search groups.
                All searchable modules that are not in ``custom_group`` will be
                grouped separately.
        """
        name2mutable: Dict[str,
                           BaseMutable] = self._build_name_mutable_mapping(
                               supernet, support_mutables)
        alias2mutable_names = self._build_alias_names_mapping(
            supernet, support_mutables)

        # Check whether the custom group is valid
        if len(custom_groups) > 0:
            self._check_valid_groups(alias2mutable_names, name2mutable,
                                     custom_groups)

        # Construct search_groups based on user-defined group
        search_groups: Dict[int, List[BaseMutable]] = dict()

        current_group_nums = 0
        grouped_mutable_names: List[str] = list()
        grouped_alias: List[str] = list()
        for group in custom_groups:
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
            if isinstance(module, support_mutables):
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

        return search_groups

    def _check_valid_groups(self, alias2mutable_names: Dict[str, List[str]],
                            name2mutable: Dict[str, BaseMutable],
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


class MutatorProtocol(Protocol):  # pragma: no cover

    @property
    def mutable_class_type(self) -> Type[BaseMutable]:
        ...

    @property
    def search_groups(self) -> Dict:
        ...


class OneShotSampleMixin:

    def sample_choices(self: MutatorProtocol) -> Dict:
        random_choices = dict()
        for group_id, modules in self.search_groups.items():
            random_choices[group_id] = modules[0].sample_choice()

        return random_choices

    def set_choices(self: MutatorProtocol, choices: Dict) -> None:
        for group_id, modules in self.search_groups.items():
            choice = choices[group_id]
            for module in modules:
                module.current_choice = choice


class DynamicSampleMixin(OneShotSampleMixin):

    @property
    def max_choices(self: MutatorProtocol) -> Dict:
        max_choices = dict()
        for group_id, modules in self.search_groups.items():
            max_choices[group_id] = modules[0].max_choice

        return max_choices

    @property
    def min_choices(self: MutatorProtocol) -> Dict:
        min_choices = dict()
        for group_id, modules in self.search_groups.items():
            min_choices[group_id] = modules[0].min_choice

        return min_choices
