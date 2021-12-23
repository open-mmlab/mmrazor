# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from functools import partial

from torch import nn

from mmrazor.models.builder import MUTATORS
from mmrazor.models.mutables import MutableModule
from .base import BaseMutator


@MUTATORS.register_module()
class DifferentiableMutator(BaseMutator):
    """A mutator for the differentiable NAS, which mainly provide some core
    functions of changing the structure of ``ARCHITECTURES``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_from_supernet(self, supernet):
        """Inherit from ``BaseMutator``'s, execute some customized functions
        exclude implementing origin ``prepare_from_supernet``.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """
        super().prepare_from_supernet(supernet)
        self.arch_params = self.build_arch_params(supernet)
        self.modify_supernet_forward(supernet)

    def build_arch_params(self, supernet):
        """This function will build many arch params, which are generally used
        in diffirentiale search algorithms, such as Darts' series. Each
        space_id corresponds to an arch param, so the Mutable with the same
        space_id share the same arch param.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.

        Returns:
            torch.nn.ParameterDict: the arch params are got after traversing
                the supernet.
        """

        arch_params = nn.ParameterDict()

        # Traverse all the child modules of the model. If a child module is an
        # Space instance and its space_id is not recorded, call its
        # :func:'build_space_architecture' and record the return value. If not,
        # pass.
        def traverse(module):
            for name, child in module.named_children():

                if isinstance(child, MutableModule):
                    space_id = child.space_id
                    if space_id not in arch_params:
                        space_arch_param = child.build_arch_param()
                        if space_arch_param is not None:
                            arch_params[space_id] = space_arch_param

                traverse(child)

        traverse(supernet)

        return arch_params

    def modify_supernet_forward(self, supernet):
        """Modify the supernet's default value in forward. Traverse all child
        modules of the model, modify the supernet's default value in
        :func:'forward' of each Space.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """

        def traverse(module):
            for name, child in module.named_children():

                if isinstance(child, MutableModule):
                    if child.space_id in self.arch_params.keys():
                        space_id = child.space_id
                        space_arch_param = self.arch_params[space_id]
                        child.forward = partial(
                            child.forward, arch_param=space_arch_param)

                traverse(child)

        traverse(supernet)

    @abstractmethod
    def search_subnet(self):
        pass
