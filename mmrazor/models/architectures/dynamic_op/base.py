# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn

from mmrazor.models.mutables.mutable_channel import MutableChannel


class DynamicOP(ABC):

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        ...


class ChannelDynamicOP(DynamicOP):

    @property
    @abstractmethod
    def mutable_in(self) -> MutableChannel:
        ...

    @property
    def mutable_out(self) -> MutableChannel:
        ...
