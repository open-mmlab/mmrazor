# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from ...mutables import DiffMutableModule
from .module_mutator import ModuleMutator


@MODELS.register_module()
class DiffModuleMutator(ModuleMutator):
    """Differentiable mutable based mutator.

    `DiffModuleMutator` is responsible for mutating `DiffMutableModule`,
    `DiffMutableOP`, `DiffChoiceRoute` and `GumbelChoiceRoute`.
    The architecture parameters of the mutables are retained
    in `DiffModuleMutator`.

    Args:
        custom_group (list[list[str]], optional): User-defined search groups.
            All searchable modules that are not in ``custom_group`` will be
            grouped separately.
    """

    def __init__(self,
                 custom_group: Optional[List[List[str]]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(custom_group=custom_group, init_cfg=init_cfg)

    def prepare_from_supernet(self,
                              supernet: nn.Module,
                              is_random: bool = True) -> None:
        """Inherit from ``BaseMutator``'s, generate `arch_params` in DARTS.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

        super().prepare_from_supernet(supernet)
        self.arch_params = self.build_arch_params(is_random=is_random)
        self.modify_supernet_forward(self.arch_params)

    def build_arch_params(self, is_random: bool = True):
        """This function will build many arch params, which are generally used
        in differentiable search algorithms, such as Darts' series. Each
        group_id corresponds to an arch param, so the Mutables with the same
        group_id share the same arch param.

        Returns:
            torch.nn.ParameterDict: the arch params are got by `search_groups`.
        """

        arch_params = nn.ParameterDict()

        for group_id, modules in self.search_groups.items():
            group_arch_param = modules[0].build_arch_param(is_random)
            arch_params[str(group_id)] = group_arch_param

        return arch_params

    def modify_supernet_forward(self, arch_params):
        """Modify the DiffMutableModule's default arch_param in forward.

        In MMRazor, the `arch_param` is along to `DiffModuleMutator`, while the
        `DiffMutableModule` needs `arch_param` in the forward. Here we use
        partial function to assign the corresponding `arch_param` to each
        `DiffMutableModule`.
        """

        for group_id, mutables in self.search_groups.items():
            for m in mutables:
                m.set_forward_args(arch_param=arch_params[str(group_id)])

    def sample_choices(self):
        """Sampling by search groups.

        The sampling result of the first mutable of each group is the sampling
        result of this group.

        Returns:
            Dict[int, Any]: Random choices dict.
        """

        choices = dict()
        for group_id, mutables in self.search_groups.items():
            arch_parm = self.arch_params[str(group_id)]
            choice = mutables[0].sample_choice(arch_parm)
            choices[group_id] = choice
        return choices

    def set_choices(self, choices: Dict[int, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.

        Args:
            choices (Dict[int, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for group_id, mutables in self.search_groups.items():
            choice = choices[group_id]
            for m in mutables:
                m.current_choice = choice

    def sample_weights(self,
                       arch_param: nn.Parameter,
                       cate_prob: torch.Tensor,
                       random_sample: bool = False):
        """Use one-hot distributions to sample the arch weights based on the
        arch params."""
        if random_sample:
            uni = torch.ones_like(arch_param)
            m = D.one_hot_categorical.OneHotCategorical(uni)
        else:
            m = D.one_hot_categorical.OneHotCategorical(probs=cate_prob)
        return m.sample()

    @property
    def mutable_class_type(self):
        """Differentiable mutable class type.

        Returns:
            Type[DiffMutableModule]: Class type of differentiable mutable.
        """
        return DiffMutableModule

    def compute_loss(self):
        """Compute mutator loss."""
        mutator_loss = 0.0
        self.arch_weights = dict()
        for k, v in self.arch_params.items():
            probs = F.softmax(v, -1)
            self.arch_weights[k] = self.sample_weights(v, probs)
            self.arch_weights[k].requires_grad_()
            mutator_loss += torch.log(
                (self.arch_weights[k] * probs).sum(-1)).sum()
        return mutator_loss

    def handle_grads(self):
        """Handle grads of arch params & arch weights."""
        assert hasattr(self, 'arch_weights'), \
            'Call self.gather_loss() first to get self.arch_weights.'
        for k, v in self.arch_params.items():
            v.grad.data.mul_(self.arch_weights[k].grad.data.sum())
            self.arch_weights[k].grad.zero_()
