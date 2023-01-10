# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn

from mmrazor.models.mutables import BaseMutable
from mmrazor.models.mutators import DiffModuleMutator


class SpaceMixin():
    """A mixin for :class:`BaseAlgorithm`, which can unify ``search_groups``
    from different mutators into the overall search_space with providing APIs
    for subnet operations.

    Note:
        SpaceMixin is designed for handling different type of mutators,
        therefore it currently supports NAS/Pruning algorithms with mutator(s).
    """

    def _build_search_space(self, prefix=''):
        """Unify search_groups from different type of mutators as an overall
        search_space of the model."""
        assert hasattr(self, 'mutators') or hasattr(self, 'mutator'), (
            '_build_search_space is only suitable for algorithms with mutator.'
        )
        self.search_space = dict()
        if hasattr(self, 'mutators'):
            for name, mutator in self.mutators.items():
                assert hasattr(mutator, 'search_groups'), (
                    '`mutator.prepare_from_supernet()` must be called before '
                    'building the overall search space.')

                prefix = name.split('_')[0] + '_'
                for k, v in mutator.search_groups.items():
                    self.search_space[prefix + str(k)] = v
        else:
            assert hasattr(self.mutator, 'search_groups'), (
                '`mutator.prepare_from_supernet()` must be called before '
                'building the overall search space.')

            for name in ['Value', 'Channel', 'Module']:
                if name in self.mutator.__class__.__name__:
                    prefix = name.lower() + '_'
                    break
            for k, v in self.mutator.search_groups.items():
                self.search_space[prefix + str(k)] = v

            if isinstance(self.mutator, DiffModuleMutator):
                self.search_params = self.build_search_params()
                self.modify_supernet_forward()

    def build_search_params(self):
        """This function will build searchable params for each layer, which are
        generally used in differentiable search algorithms, such as Darts'
        series. Each name corresponds to an search param, so the Mutables with
        the same name share the same search param.

        Returns:
            torch.nn.ParameterDict: search params got by the search_space.
        """
        search_params = nn.ParameterDict()

        for name, mutables in self.search_space.items():
            search_params[name] = nn.Parameter(
                torch.randn(mutables[0].num_choices) * 1e-3)

        return search_params

    def modify_supernet_forward(self):
        """Modify the DiffMutableModule's default arch_param in forward.

        In MMRazor, the `DiffMutableModule` needs `arch_param` in the forward.
        Here we use partial function to assign the corresponding `arch_param`
        to each `DiffMutableModule`.
        """
        for name, mutables in self.search_space.items():
            for mutable in mutables:
                mutable.set_forward_args(arch_param=self.search_params[name])

    def sample_subnet(self, kind='random') -> Dict:
        """Random sample subnet by search space."""
        subnet = dict()
        for name, mutables in self.search_space.items():
            if hasattr(self, 'mutator') and \
               isinstance(self.mutator, DiffModuleMutator):  # type: ignore
                search_param = self.search_params[name]
                subnet[name] = mutables[0].sample_choice(search_param)
            else:
                if kind == 'max':
                    subnet[name] = mutables[0].max_choice
                elif kind == 'min':
                    subnet[name] = mutables[0].min_choice
                else:
                    subnet[name] = mutables[0].sample_choice()
        return subnet

    def set_subnet(self, choices: Dict) -> None:
        """Set choices for each mutable in search space."""
        for name, mutables in self.search_space.items():
            if name not in choices:
                if mutables[0].alias not in choices:
                    # allow optional target_prune_ratio
                    continue
                else:
                    choice = choices[mutables[0].alias]
            else:
                choice = choices[name]

            for mutable in mutables:
                mutable.current_choice = choice

    def fix_subnet(self):
        """Fix subnet."""
        self.set_subnet(self.sample_subnet())
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(module.current_choice)
        self.is_supernet = False

    @property
    def max_subnet(self) -> Dict:
        """Get max choices for each mutable in search space."""
        max_subnet = dict()
        for name, mutables in self.search_space.items():
            max_subnet[name] = mutables[0].max_choice

        return max_subnet

    @property
    def min_subnet(self) -> Dict:
        """Get min choices for each mutable in search space."""
        min_subnet = dict()
        for name, mutables in self.search_space.items():
            min_subnet[name] = mutables[0].min_choice

        return min_subnet

    @property
    def current_subnet(self) -> Dict:
        """Get current subnet by search space."""
        current_subnet = dict()
        for name, mutables in self.search_space.items():
            current_subnet[name] = mutables[0].current_choice

        return current_subnet

    def set_max_subnet(self):
        """Set max choices for each mutable in search space."""
        for name, mutables in self.search_space.items():
            choice = self.max_subnet[name]
            for mutable in mutables:
                mutable.current_choice = choice

    def set_min_subnet(self):
        """Set min choices for each mutable in search space."""
        for name, mutables in self.search_space.items():
            choice = self.min_subnet[name]
            for mutable in mutables:
                mutable.current_choice = choice
