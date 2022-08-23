# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Optional, Set

from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable


class DynamicOP(ABC):
    """Base class for dynamic OP. A dynamic OP usually consists of a normal
    static OP and mutables, where mutables are used to control the searchable
    (mutable) part of the dynamic OP.

    Note:
        When the dynamic OP has just been initialized, its forward propagation
        logic should be the same as the corresponding static OP. Only after
        the searchable part accepts the specific mutable through the
        corresponding interface does the part really become dynamic.

    Note:
        All subclass should implement ``to_static_op`` API.

    Args:
        accepted_mutables (set): The string set of all accepted mutables.
    """
    accepted_mutables: Set[str] = set()

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        """Convert dynamic OP to static OP.

        Note:
            The forward result for the same input between dynamic OP and its
            corresponding static OP must be same.

        Returns:
            nn.Module: Corresponding static OP.
        """

    def check_if_mutables_fixed(self) -> None:
        """Check if all mutables are fixed.

        Raises:
            RuntimeError: Error if a existing mutable is not fixed.
        """

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable {type(mutable)} is not fixed.')

        for mutable in self.accepted_mutables:
            check_fixed(getattr(self, f'{mutable}'))

    @staticmethod
    def get_current_choice(mutable: BaseMutable) -> Any:
        """Get current choice of given mutable.

        Args:
            mutable (BaseMutable): Given mutable.

        Raises:
            RuntimeError: Error if `current_choice` is None.

        Returns:
            Any: Current choice of given mutable.
        """
        current_choice = mutable.current_choice
        if current_choice is None:
            raise RuntimeError(f'current choice of mutable {type(mutable)} '
                               'can not be None at runtime')

        return current_choice


class ChannelDynamicOP(DynamicOP):
    """Base class for dynamic OP with mutable channels.

    Note:
        All subclass should implement ``mutable_in`` and ``mutable_out`` APIs.
    """

    @property
    @abstractmethod
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable related to input."""

    @property
    @abstractmethod
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable related to output."""

    @staticmethod
    def check_mutable_channels(mutable_channels: BaseMutable) -> None:
        """Check if mutable has `currnet_mask` attribute.

        Args:
            mutable_channels (BaseMutable): Mutable to be checked.

        Raises:
            ValueError: Error if mutable does not have `current_mask`
                attribute.
        """
        if not hasattr(mutable_channels, 'current_mask'):
            raise ValueError(
                'channel mutable must have attribute `current_mask`')
