# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmcv.runner import BaseModule

from mmrazor.registry import MODELS


@MODELS.register_module()
class SlimmableChannelMutable(BaseModule):
    """A type of ``MUTABLES`` to train several subnet together, such as the
    retraining stage in AutoSlim.

    Notes:
        We need to set `candidate_choices` after the instantiation of a
        `SlimmableChannelMutable` by ourselves.

    Args:
        name (str): Mutable name.
        mask_type (str): One of 'in_mask' or 'out_mask'.
        num_channels (int): The raw number of channels.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 name: str,
                 mask_type: str,
                 num_channels: int,
                 init_cfg: Optional[Dict] = None):
        super(SlimmableChannelMutable, self).__init__(init_cfg=init_cfg)

        self.name = name
        assert mask_type in ('in_mask', 'out_mask')
        self.mask_type = mask_type
        self.num_channels = num_channels
        self.register_buffer('_mask', torch.ones((num_channels, )).bool())
        self._current_choice = 0

    @property
    def candidate_choices(self) -> List:
        """A list of candidate channel numbers."""
        return self._candidate_choices

    @candidate_choices.setter
    def candidate_choices(self, choices):
        """Set the candidate channel numbers."""
        assert getattr(self, '_candidate_choices', None) is None, \
            f'candidate_choices can be set only when candidate_choices is ' \
            f'None, got: candidate_choices = {self._candidate_choices}'

        assert all([num > 0 and num <= self.num_channels
                    for num in choices]), \
            f'The candidate channel numbers should be in ' \
            f'range(0, {self.num_channels}].'
        assert all([isinstance(num, int) for num in choices]), \
            'Type of `candidate_choices` should be int.'

        self._candidate_choices = list(choices)

    @property
    def choices(self) -> List:
        """Return all subnet indexes."""
        assert self._candidate_choices is not None
        return list(range(len(self.candidate_choices)))

    @property
    def current_choice(self) -> int:
        """The current choice of the mutable."""
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: int):
        """Set the current choice of the mutable."""
        assert choice in self.choices
        self._current_choice = choice

    @property
    def mask(self):
        """The current mask.

        We slice the registered parameters and buffers of a ``nn.Module``
        according to the mask of the corresponding channel mutable.
        """
        idx = self.current_choice
        num_channels = self.candidate_choices[idx]
        mask = torch.zeros_like(self._mask).bool()
        mask[:num_channels] = True
        return mask
