# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.nn import functional as F

from mmrazor.models.builder import MUTATORS
from .differentiable_mutator import DifferentiableMutator


@MUTATORS.register_module()
class DartsMutator(DifferentiableMutator):

    def __init__(self, ignore_choices=('zero', ), **kwargs):
        super().__init__(**kwargs)
        self.ignore_choices = ignore_choices

    def search_subnet(self):

        subnet_dict = dict()
        for space_id, sub_space in self.search_spaces.items():
            if space_id in self.arch_params:
                space_arch_param = self.arch_params[space_id]
                arch_probs = F.softmax(space_arch_param, dim=-1)
                choice_names = sub_space['choice_names']
                keep_idx = [
                    i for i, name in enumerate(choice_names)
                    if name not in self.ignore_choices
                ]
                best_choice_prob, best_choice_idx = torch.max(
                    arch_probs[keep_idx], 0)
                best_choice_idx = keep_idx[best_choice_idx.item()]
                best_choice_name = choice_names[best_choice_idx]

                subnet_dict[space_id] = dict(
                    chosen=[best_choice_name],
                    chosen_probs=[best_choice_prob.item()])

        def sort_key(x):
            return subnet_dict[x]['chosen_probs'][0]

        for space_id, sub_space in self.search_spaces.items():
            if space_id not in self.arch_params:
                num_chosen = sub_space['num_chosen']
                choice_names = sub_space['choice_names']
                sorted_edges = list(
                    sorted(choice_names, key=sort_key, reverse=True))
                chosen = sorted_edges[:num_chosen]
                subnet_dict[space_id] = dict(chosen=chosen)

                for not_chosen in sorted_edges[num_chosen:]:
                    subnet_dict.pop(not_chosen)

        return subnet_dict
