# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable

MUTABLE_TYPE = BaseMutable
MUTABLES_TYPE = Dict[str, MUTABLE_TYPE]
OPT_MUTABLES_TYPE = Optional[Dict[str, MUTABLE_TYPE]]


class DynamicOP(ABC):
    accepted_mutable_keys: Set[str] = set()

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        ...

    def check_if_mutables_fixed(self) -> None:

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable {type(mutable)} is not fixed.')

        for mutable_key in self.accepted_mutable_keys:
            check_fixed(getattr(self, f'{mutable_key}_mutable'))

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
    def check_channels_mutable(channels_mutable: BaseMutable) -> None:
        if not hasattr(channels_mutable, 'current_mask'):
            raise ValueError(
                'channel mutable must have attribute `current_mask`')
