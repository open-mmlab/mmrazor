# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule


class MutableModule(BaseModule, metaclass=ABCMeta):
    """Base class for ``MUTABLES``. Searchable module for building searchable
    architecture in NAS. It mainly consists of module and mask, and achieving
    searchable function by handling mask.

    Args:
        space_id (str): Used to index ``Placeholder``, it is one and only index
            for each ``Placeholder``.
        num_chosen (str): The number of chosen ``OPS`` in the ``MUTABLES``.
        init_cfg (dict): Init config for ``BaseModule``.
    """

    def __init__(self, space_id, num_chosen=1, init_cfg=None, **kwargs):
        super(MutableModule, self).__init__(init_cfg)
        self.space_id = space_id
        self.num_chosen = num_chosen

    @abstractmethod
    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    @abstractmethod
    def build_choices(self, cfg):
        """Build all chosen ``OPS`` used to combine ``MUTABLES``, and the
        choices will be sampled.

        Args:
            cfg (dict): The config for the choices.
        """
        pass

    def build_choice_mask(self):
        """Generate the choice mask for the choices of ``MUTABLES``.

        Returns:
            torch.Tensor: Init choice mask. Its elements' type is bool.
        """
        if torch.cuda.is_available():
            return torch.ones(self.num_choices).bool().cuda()
        else:
            return torch.ones(self.num_choices).bool()

    def set_choice_mask(self, mask):
        """Use the mask to update the choice mask.

        Args:
            mask (torch.Tensor): Choice mask specified to update the choice
                mask.
        """
        assert self.choice_mask.size(0) == mask.size(0)
        self.choice_mask = mask

    @property
    def num_choices(self):
        """The number of the choices.

        Returns:
            int: the length of the choices.
        """
        return len(self.choices)

    @property
    def choice_names(self):
        """The choices' names.

        Returns:
            tuple: The keys of the choices.
        """
        assert isinstance(self.choices, nn.ModuleDict), \
            'candidates must be nn.ModuleDict.'
        return tuple(self.choices.keys())

    @property
    def choice_modules(self):
        """The choices' modules.

        Returns:
            tuple: The values of the choices.
        """
        assert isinstance(self.choices, nn.ModuleDict), \
            'candidates must be nn.ModuleDict.'
        return tuple(self.choices.values())

    def build_space_mask(self):
        """Generate the space mask for the search spaces of ``MUTATORS``.

        Returns:
            torch.Tensor: Init choice mask. Its elements' type is float.
        """
        if torch.cuda.is_available():
            return torch.ones(self.num_choices).cuda() * 1.0
        else:
            return torch.ones(self.num_choices) * 1.0

    def export(self, chosen):
        """Delete not chosen ``OPS`` in the choices.

        Args:
            chosen (list[str]): Names of chosen ``OPS``.
        """
        for name in self.choice_names:
            if name not in chosen:
                self.choices.pop(name)
