# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable

MUTABLE_TYPE = BaseMutable
MUTABLES_TYPE = Dict[str, MUTABLE_TYPE]
OPT_MUTABLES_TYPE = Optional[Dict[str, MUTABLE_TYPE]]


class DynamicOP(ABC):
    accpeted_mutables: Set[str] = set()

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        ...

    def check_if_mutables_fixed(self) -> None:

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable {type(mutable)} is not fixed.')

        for mutable in self.accpeted_mutables:
            check_fixed(getattr(self, f'{mutable}'))

    @staticmethod
    def get_current_choice(mutable: BaseMutable) -> Any:
        current_choice = mutable.current_choice
        if current_choice is None:
            raise RuntimeError(f'current choice of mutable {type(mutable)} '
                               'can not be None at runtime')

        return current_choice


class ChannelDynamicOP(DynamicOP):

    @property
    @abstractmethod
    def mutable_in(self) -> Optional[BaseMutable]:
        ...

    @property
    @abstractmethod
    def mutable_out(self) -> Optional[BaseMutable]:
        ...

    @staticmethod
    def check_mutable_channels(mutable_channels: BaseMutable) -> None:
        if not hasattr(mutable_channels, 'current_mask'):
            raise ValueError(
                'channel mutable must have attribute `current_mask`')
