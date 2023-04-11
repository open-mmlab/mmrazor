# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from typing import Dict, List

from torch.nn import Module

from mmrazor.models.mutables import MutableValue
from mmrazor.models.mutables.mutable_module import MutableModule
from .base_mutator import MUTABLE_TYPE


class GroupMixin():
    """A mixin for :class:`BaseMutator`, which can group mutables by
    ``custom_group`` and ``alias``(see more information in
    :class:`MUTABLE_TYPE`). Grouping by alias and module name are both
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

    def is_supported_mutable(self, module):
        """Judge whether is a supported mutable."""
        for mutable_type in [MutableModule, MutableValue]:
            if isinstance(module, mutable_type):
                return True
        return False

    def _build_name_mutable_mapping(
            self, supernet: Module) -> Dict[str, MUTABLE_TYPE]:
        """Mapping module name to mutable."""
        name2mutable: Dict[str, MUTABLE_TYPE] = dict()
        for name, module in supernet.named_modules():
            if self.is_supported_mutable(module):
                name2mutable[name] = module
            elif hasattr(module, 'source_mutables'):
                for each_mutable in module.source_mutables:
                    if self.is_supported_mutable(each_mutable):
                        name2mutable[name] = each_mutable

        self._name2mutable = name2mutable

        return name2mutable

    def _build_alias_names_mapping(self,
                                   supernet: Module) -> Dict[str, List[str]]:
        """Mapping alias to module names."""
        alias2mutable_names: Dict[str, List[str]] = dict()

        def _append(key, dict, name):
            if key not in dict:
                dict[key] = [name]
            else:
                dict[key].append(name)

        for name, module in supernet.named_modules():
            if self.is_supported_mutable(module):
                if module.alias is not None:
                    _append(module.alias, alias2mutable_names, name)
            elif hasattr(module, 'source_mutables'):
                for each_mutable in module.source_mutables:
                    if self.is_supported_mutable(each_mutable):
                        if each_mutable.alias is not None:
                            _append(each_mutable.alias, alias2mutable_names,
                                    name)

        return alias2mutable_names

    def build_search_groups(
            self, supernet: Module,
            custom_groups: List[List[str]]) -> Dict[str, List[MUTABLE_TYPE]]:
        """Build search group with ``custom_group`` and ``alias``(see more
        information in :class:`MUTABLE_TYPE`). Grouping by alias and module
        name are both supported.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
            support_mutables (Type): Mutable type that can be grouped.
            custom_group (list, optional): User-defined search groups.
                All searchable modules that are not in ``custom_group`` will be
                grouped separately.

        Return:
            search_groups (Dict[str, List[MUTABLE_TYPE]]): The built
                search_groups.
        """
        name2mutable: Dict[
            str, MUTABLE_TYPE] = self._build_name_mutable_mapping(supernet)
        alias2mutable_names = self._build_alias_names_mapping(supernet)

        # Check whether the custom group is valid
        if len(custom_groups) > 0:
            self._check_valid_groups(alias2mutable_names, name2mutable,
                                     custom_groups)

        # Construct search_groups based on user-defined group
        search_groups: Dict[str, List[MUTABLE_TYPE]] = dict()

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

            # TODO: fix prefix when constructing custom groups.
            prefix = name2mutable[item].mutable_prefix
            group_name = prefix + '_' + str(current_group_nums)
            search_groups[group_name] = group_mutables
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
                    prefix = name2mutable[mutable_names[0]].mutable_prefix
                    group_name = prefix + '_' + str(current_group_nums)
                    search_groups[group_name] = []
                    for mutable_name in mutable_names:
                        if mutable_name not in grouped_mutable_names:
                            search_groups[group_name].append(
                                name2mutable[mutable_name])
                            grouped_mutable_names.append(mutable_name)
                    current_group_nums += 1

        # check whether all the mutable objects are in the search_groups
        for name, module in supernet.named_modules():
            if self.is_supported_mutable(module):
                if name in grouped_mutable_names:
                    continue
                else:
                    prefix = module.mutable_prefix
                    group_name = prefix + '_' + str(current_group_nums)
                    search_groups[group_name] = [module]
                    current_group_nums += 1
            elif hasattr(module, 'source_mutables'):
                for each_mutable in module.source_mutables:
                    if self.is_supported_mutable(each_mutable):
                        if name in grouped_mutable_names:
                            continue
                        else:
                            prefix = each_mutable.mutable_prefix
                            group_name = prefix + '_' + str(current_group_nums)
                            search_groups[group_name] = [each_mutable]
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
                            name2mutable: Dict[str, MUTABLE_TYPE],
                            custom_group: List[List[str]]) -> None:
        """Check if all keys are legal."""
        aliases = [*alias2mutable_names.keys()]
        module_names = [*name2mutable.keys()]

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
