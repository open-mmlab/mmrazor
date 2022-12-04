# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict

import torch.nn as nn
from mmengine import print_log
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ...ops import ShortcutLayer
from ..mixins import DynamicChannelMixin


@MODELS.register_module()
class DynamicShortcutLayer(ShortcutLayer, DynamicChannelMixin):

    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def mutable_in_channels(self):
        """Mutable in_channels."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['in_channels']

    @property
    def mutable_out_channels(self):
        """Mutable out_channels."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['out_channels']

    @classmethod
    def convert_from(cls, module):
        """Convert the static module to dynamic one."""
        shortcut = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
        )
        return shortcut

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        """Register attribute of mutable."""
        self.check_mutable_attr_valid(attr)
        if attr in self.attr_mappings:
            attr_map = self.attr_mappings[attr]
            assert attr_map in self.accepted_mutable_attrs
            if attr_map in self.mutable_attrs:
                print_log(
                    f'{attr_map}({attr}) is already in `mutable_attrs`',
                    level=logging.WARNING)
            else:
                self._register_mutable_attr(attr_map, mutable)
        elif attr in self.accepted_mutable_attrs:
            self._register_mutable_attr(attr, mutable)
        else:
            raise NotImplementedError

    def _register_mutable_attr(self, attr: str, mutable: BaseMutable):
        """Register `in_channels` `out_channels`"""
        if attr == 'in_channels':
            self._register_mutable_in_channels(mutable)
        elif attr == 'out_channels':
            self._register_mutable_out_channels(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_in_channels(self, mutable_in_channels):
        """Register the mutable number of heads."""
        assert hasattr(self, 'mutable_attrs')

        mask_size = mutable_in_channels.current_mask.size(0)
        if mask_size > self.in_channels:
            raise ValueError(
                f'Expect mask size of mutable to be {self.in_channels} as '
                f'`in_channels`, but got: {mask_size}.')

        self.mutable_attrs['in_channels'] = mutable_in_channels

    def _register_mutable_out_channels(self, mutable_out_channels):
        """Register mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')

        mask_size = mutable_out_channels.current_mask.size(0)
        if mask_size != self.out_channels:
            raise ValueError(
                f'Expect mask size of mutable to be {self.out_channels} as '
                f'`out_channels`, but got: {mask_size}.')

        self.mutable_attrs['out_channels'] = mutable_out_channels

    def _get_dynamic_params(self: ShortcutLayer):
        """Get mask of ``ShortcutLayer``"""
        mutable_in_channels = self.conv.mutable_in_channels
        mutable_out_channels = self.conv.mutable_out_channels
        return mutable_in_channels, mutable_out_channels

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return ShortcutLayer

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        return self.static_op_factory(
            in_channels=self.conv.mutable_in_channels,
            out_channels=self.conv.mutable_out_channels)  # type:ignore

    def forward(self, x: Tensor) -> Tensor:
        mutable_in_channels, mutable_out_channels = self._get_dynamic_params()

        self.in_channels = mutable_in_channels.current_mask.sum().item()
        self.out_channels = mutable_out_channels.current_mask.sum().item()

        return super().forward(x)
