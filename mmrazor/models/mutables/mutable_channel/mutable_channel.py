# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List

import torch

from ..base_mutable import CHOICE_TYPE, CHOSEN_TYPE, BaseMutable
from ..derived_mutable import DerivedMethodMixin


class MutableChannel(BaseMutable[CHOICE_TYPE, CHOSEN_TYPE],
                     DerivedMethodMixin):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot channel mutables.

    Args:
        num_channels (int): The raw number of channels.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self._same_mutables: List[MutableChannel] = list()

        # If the input of a module is a concatenation of several modules'
        # outputs, we add the mutable out of these modules to the
        # `concat_parent_mutables` of this module.
        self.concat_parent_mutables: List[MutableChannel] = list()
        self.name = 'unbind'

    @property
    def same_mutables(self):
        """Mutables in `same_mutables` and the current mutable should change
        Synchronously."""
        return self._same_mutables

    def register_same_mutable(self, mutable):
        """Register the input mutable in `same_mutables`."""
        if isinstance(mutable, list):
            # Add a concatenation of mutables to `concat_parent_mutables`.
            self.concat_parent_mutables = mutable
            return

        if self == mutable:
            return
        if mutable in self._same_mutables:
            return

        self._same_mutables.append(mutable)
        for s_mutable in self._same_mutables:
            s_mutable.register_same_mutable(mutable)
            mutable.register_same_mutable(s_mutable)

    @abstractmethod
    def convert_choice_to_mask(self, choice: CHOICE_TYPE) -> torch.Tensor:
        """Get the mask according to the input choice."""
        pass

    @property
    def current_mask(self):
        """The current mask.

        We slice the registered parameters and buffers of a ``nn.Module``
        according to the mask of the corresponding channel mutable.
        """
        if len(self.concat_parent_mutables) > 0:
            # If the input of a module is a concatenation of several modules'
            # outputs, the in_mask of this module is the concatenation of
            # these modules' out_mask.
            return torch.cat([
                mutable.current_mask for mutable in self.concat_parent_mutables
            ])
        else:
            return self.convert_choice_to_mask(self.current_choice)

    def bind_mutable_name(self, name: str) -> None:
        """Bind a MutableChannel to its name.

        Args:
            name (str): Name of this `MutableChannel`.
        """
        self.name = name

    def fix_chosen(self, chosen: CHOSEN_TYPE) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            chosen (str): The chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        self.is_fixed = True

    def __repr__(self):
        concat_mutable_name = [
            mutable.name for mutable in self.concat_parent_mutables
        ]
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'concat_mutable_name={concat_mutable_name})'
        return repr_str
