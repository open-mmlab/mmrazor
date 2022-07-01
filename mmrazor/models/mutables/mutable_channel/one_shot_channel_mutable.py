# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from mmcv.runner import BaseModule


class OneShotChannelMutable(BaseModule, ABC):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot channel mutables.

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
        super(OneShotChannelMutable, self).__init__(init_cfg=init_cfg)
        # If the input of a module is a concatenation of several modules'
        # outputs, we add the out_mutable (mask_type == 'out_mask') of
        # these modules to the `concat_mutables` of this module.
        self.concat_mutables: List[OneShotChannelMutable] = list()
        self.name = name
        assert mask_type in ('in_mask', 'out_mask')
        self.mask_type = mask_type
        self.num_channels = num_channels
        self.register_buffer('_mask', torch.ones((num_channels, )).bool())
        self._current_choice = num_channels

        self._same_mutables: List[OneShotChannelMutable] = list()

    @property
    def same_mutables(self):
        """Mutables in `same_mutables` and the current mutable should change
        Synchronously."""
        return self._same_mutables

    def register_same_mutable(self, mutable):
        """Register the input mutable in `same_mutables`."""
        if isinstance(mutable, list):
            # Add a concatenation of mutables to `concat_mutables`.
            assert self.mask_type == 'in_mask'
            assert all([
                cur_mutable.mask_type == 'out_mask' for cur_mutable in mutable
            ])
            self.concat_mutables = mutable
            return

        if self == mutable:
            return
        if mutable in self._same_mutables:
            return

        self._same_mutables.append(mutable)
        for s_mutable in self._same_mutables:
            s_mutable.register_same_mutable(mutable)
            mutable.register_same_mutable(s_mutable)

    def sample_choice(self) -> int:
        """Sample an arbitrary selection from candidate choices.

        Returns:
            int: The chosen number of channels.
        """
        assert len(self.concat_mutables) == 0
        num_channels = np.random.choice(self.choices)
        assert num_channels > 0, \
            f'Sampled number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return num_channels

    @property
    def min_choice(self) -> int:
        """Minimum number of channels."""
        assert len(self.concat_mutables) == 0
        min_channels = min(self.choices)
        assert min_channels > 0, \
            f'Minimum number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return min_channels

    @property
    def max_choice(self) -> int:
        """Maximum number of channels."""
        return max(self.choices)

    def get_choice(self, idx: int) -> int:
        """Get the `idx`-th choice from candidate choices."""
        assert len(self.concat_mutables) == 0
        num_channels = self.choices[idx]
        assert num_channels > 0, \
            f'Number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return num_channels

    @property
    def current_choice(self):
        """The current choice of the mutable."""
        if len(self.concat_mutables) > 0:
            return sum(
                [mutable.current_choice for mutable in self.concat_mutables])
        else:
            return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: int):
        """Set the current choice of the mutable."""
        assert choice in self.choices
        self._current_choice = choice

    @property
    @abstractmethod
    def choices(self) -> List[int]:
        """list: all choices. """

    @property
    def mask(self):
        """The current mask.

        We slice the registered parameters and buffers of a ``nn.Module``
        according to the mask of the corresponding channel mutable.
        """
        if len(self.concat_mutables) > 0:
            # If the input of a module is a concatenation of several modules'
            # outputs, the in_mask of this module is the concatenation of
            # these modules' out_mask.
            return torch.cat(
                [mutable.mask for mutable in self.concat_mutables])
        else:
            num_channels = self.current_choice
            mask = torch.zeros_like(self._mask).bool()
            mask[:num_channels] = True
            return mask

    def __repr__(self):
        concat_mutable_name = [
            mutable.name for mutable in self.concat_mutables
        ]
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'mask_type={self.mask_type}, '
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'concat_mutable_name={concat_mutable_name})'
        return repr_str
