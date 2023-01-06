# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict


class SpaceMixin():

    def _build_search_space(self, prefix=''):
        """Unify search_groups in mutators as the search_space of the model."""
        assert hasattr(self, 'mutators') or hasattr(self, 'mutator'), (
            '_trace_search_space is only suitable for algorithms with mutator.'
        )
        self.search_space = dict()
        if hasattr(self, 'mutators'):
            for name, mutator in self.mutators.items():
                prefix = name.split('_')[0] + '_'
                for k, v in mutator.search_groups.items():
                    self.search_space[prefix + str(k)] = v
        else:
            for name in ['Value', 'Channel', 'Module']:
                if name in self.mutator.__class__.__name__:
                    prefix = name.lower() + '_'
                    break
            for k, v in self.mutator.search_groups.items():
                self.search_space[prefix + str(k)] = v

    def sample_subnet(self, kind='random') -> Dict:
        """Random sample subnet by mutator."""
        subnet = dict()
        for name, modules in self.search_space.items():
            if kind == 'max':
                subnet[name] = modules[0].max_choice
            elif kind == 'min':
                subnet[name] = modules[0].min_choice
            else:
                subnet[name] = modules[0].sample_choice()
        return subnet

    def set_subnet(self, choices: Dict) -> None:
        """Set choices for each module in search space."""
        for group_id, modules in self.search_space.items():
            if group_id not in choices:
                # allow optional target_prune_ratio
                continue
            choice = choices[group_id]
            for module in modules:
                module.current_choice = choice

    @property
    def max_subnet(self) -> Dict:
        """Get max choices for each module in search space."""
        max_subnet = dict()
        for group_id, modules in self.search_space.items():
            max_subnet[group_id] = modules[0].max_choice

        return max_subnet

    @property
    def min_subnet(self) -> Dict:
        """Get min choices for each module in search space."""
        min_subnet = dict()
        for group_id, modules in self.search_space.items():
            min_subnet[group_id] = modules[0].min_choice

        return min_subnet

    @property
    def current_subnet(self) -> Dict:
        """Get current subnet."""
        current_subnet = dict()
        for group_id, modules in self.search_space.items():
            current_subnet[group_id] = modules[0].current_choice

        return current_subnet

    def set_max_subnet(self):
        """Set max choices for each module in search space."""
        for group_id, modules in self.search_space.items():
            choice = self.max_subnet[group_id]
            for module in modules:
                module.current_choice = choice

    def set_min_subnet(self):
        """Set min choices for each module in search space."""
        for group_id, modules in self.search_space.items():
            choice = self.min_subnet[group_id]
            for module in modules:
                module.current_choice = choice
