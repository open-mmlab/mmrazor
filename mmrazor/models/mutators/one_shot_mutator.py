# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial

import numpy as np
import torch
import torch.distributed as dist

from mmrazor.models.builder import MUTATORS
from .base import BaseMutator


@MUTATORS.register_module()
class OneShotMutator(BaseMutator):
    """A mutator for the one-shot NAS, which mainly provide some core functions
    of changing the structure of ``ARCHITECTURES``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_random_mask(space_info, searching):
        """Generate random mask for randomly sampling.

        Args:
            space_info (dict): Record the information of the space need
                to sample.
            searching (bool): Whether is in search stage.

        Returns:
            torch.Tensor: Random mask generated.
        """
        space_mask = space_info['space_mask']
        num_chosen = space_info['num_chosen']
        assert num_chosen <= space_mask.size()[0]
        choice_idx = torch.multinomial(space_mask, num_chosen)
        choice_mask = torch.zeros_like(space_mask)
        choice_mask[choice_idx] = 1
        if dist.is_available() and dist.is_initialized() and not searching:
            dist.broadcast(choice_mask, src=0)
        return choice_mask

    def sample_subnet(self, searching=False):
        """Random sample subnet by random mask.

        Args:
            searching (bool): Whether is in search stage.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are random mask generated.
        """
        subnet_dict = dict()
        for space_id, space_info in self.search_spaces.items():
            subnet_dict[space_id] = self.get_random_mask(space_info, searching)
        return subnet_dict

    def set_subnet(self, subnet_dict):
        """Setting subnet in the supernet based on the result of
        ``sample_subnet`` by changing the flag: ``in_subnet``, which is easy to
        implement some operations for subnet, such as ``forward``, calculate
        flops and so on.

        Args:
            subnet_dict (dict): Record the information to build the subnet
                from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are masks.
        """
        for space_id, space_info in self.search_spaces.items():
            choice_mask = subnet_dict[space_id]
            for module in space_info['modules']:
                module.choice_mask = choice_mask
                for i, choice in enumerate(module.choices.values()):
                    if choice_mask[i]:
                        choice.apply(
                            partial(self.reset_in_subnet, in_subnet=True))
                    else:
                        choice.apply(
                            partial(self.reset_in_subnet, in_subnet=False))

    @staticmethod
    def reset_in_subnet(m, in_subnet=True):
        """Reset the module's attribution.

        Args:
            m (:obj:`torch.nn.Module`): The module in the supernet.
            in_subnet (bool): If the module in subnet, set ``in_subnet`` to
                True, otherwise set to False.
        """
        m.__in_subnet__ = in_subnet

    def set_chosen_subnet(self, subnet_dict):
        """Set chosen subnet in the search_spaces after searching stage.

        Args:
            subnet_dict (dict): Record the information to build the subnet from
                the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are masks.
        """
        for space_id, mask in subnet_dict.items():
            idxs = [i for i, x in enumerate(mask.tolist()) if x == 1.0]
            self.search_spaces[space_id]['chosen'] = [
                self.search_spaces[space_id]['choice_names'][i] for i in idxs
            ]

    def mutation(self, subnet_dict, prob=0.1):
        """Mutation used in evolution search.

        Args:
            subnet_dict (dict): Record the information to build the subnet
                from the supernet, its keys are the properties ``space_id``
                of placeholders in the mutator's search spaces, its values
                are masks.
            prob (float): The probability of mutation.

        Returns:
            dict: A new subnet_dict after mutation.
        """
        mutation_subnet_dict = copy.deepcopy(subnet_dict)
        for name, mask in subnet_dict.items():
            if np.random.random_sample() < prob:
                mutation_subnet_dict[name] = self.get_random_mask(
                    self.search_spaces[name], searching=True)
        return mutation_subnet_dict

    @staticmethod
    def crossover(subnet_dict1, subnet_dict2):
        """Crossover used in evolution search.

        Args:
            subnet_dict1 (dict): Record the information to build the subnet
                from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are masks.
            subnet_dict2 (dict): Record the information to build the subnet
                from the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are masks.

        Returns:
            dict: A new subnet_dict after crossover.
        """
        crossover_subnet_dict = copy.deepcopy(subnet_dict1)
        for name, mask in subnet_dict2.items():
            if np.random.random_sample() < 0.5:
                crossover_subnet_dict[name] = mask
        return crossover_subnet_dict
