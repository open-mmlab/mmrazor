# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MUTABLES
from .mutable_module import MutableModule


class MutableEdge(MutableModule):
    """Mutable Edge. In some NAS algorithms (Darts, AutoDeeplab, etc.), the
    connections between modules are searchable, such as the connections between
    a node and its previous nodes in Darts. ``MutableEdge`` has N modules to
    process N inputs respectively.

    Args:
        choices (torch.nn.ModuleDict): Unlike ``MutableOP``, there are already
            created modules in choices.
    """

    def __init__(self, choices, **kwargs):
        super(MutableEdge, self).__init__(**kwargs)
        assert isinstance(choices, nn.ModuleDict), \
            f'In {self.space_id}, the choices must be torch.nn.ModuleDict.'
        self.choices = choices
        self.choice_mask = self.build_choice_mask()
        # Once execute ``export``, the unchosen modules will be removed.
        # Record full choice names, which will be used in forward.
        self.full_choice_names = copy.deepcopy(self.choice_names)

    def build_choices(self, cfg):
        """MutableEdge's choices is already built."""
        pass


@MUTABLES.register_module()
class DifferentiableEdge(MutableEdge):
    """Differentiable Edge.

    Search the best module from choices by learnable parameters.

    Args:
        with_arch_param (bool): whether build learable architecture parameters.
    """

    def __init__(self, with_arch_param, **kwargs):
        super(DifferentiableEdge, self).__init__(**kwargs)
        self.with_arch_param = with_arch_param

    def build_arch_param(self):
        """build learnable architecture parameters."""
        if self.with_arch_param:
            return nn.Parameter(torch.randn(self.num_choices) * 1e-3)
        else:
            return None

    def compute_arch_probs(self, arch_param):
        """compute chosen probs according architecture parameters."""
        return F.softmax(arch_param, -1)

    def forward(self, prev_inputs, arch_param=None):
        """forward function.

        In some algorithms, there are several ``MutableModule`` share the same
        architecture parameters. So the architecture parameters are passed
        in as args.

        Args:
            prev_inputs (list[torch.Tensor]): each choice's inputs.
            arch_param (torch.nn.Parameter): architecture parameters.
        """

        assert len(self.full_choice_names) == len(prev_inputs), \
            f'In {self.space_id}, the length of full choice names must be \
                same as the length of previous inputs.'

        if self.with_arch_param:
            assert arch_param is not None, \
                f'In {self.space_id}, the arch_param can not be None when the \
                    with_arch_param=True.'

            # 1. compute choices' probs.
            probs = self.compute_arch_probs(arch_param)

            # 2. process each input
            outputs = list()
            for prob, module, input in zip(probs, self.choice_modules,
                                           prev_inputs):
                if prob > 0:
                    outputs.append(prob * module(input))

        else:
            outputs = list()
            for name, input in zip(self.full_choice_names, prev_inputs):
                if name not in self.choice_names:
                    continue
                module = self.choices[name]
                outputs.append(module(input))

            assert len(outputs) > 0

        return sum(outputs)


@MUTABLES.register_module()
class GumbelEdge(DifferentiableEdge):
    """Gumbel Edge.

    Search the best module from choices by gumbel trick.
    """

    def __init__(self, **kwargs):
        super(GumbelEdge, self).__init__(**kwargs)

    def set_temperature(self, temperature):
        """Modify the temperature."""
        self.temperature = temperature

    def compute_arch_probs(self, arch_param):
        """compute chosen probs by gumbel trick."""
        probs = F.gumbel_softmax(
            arch_param, tau=self.tau, hard=self.hard, dim=-1)
        return probs
