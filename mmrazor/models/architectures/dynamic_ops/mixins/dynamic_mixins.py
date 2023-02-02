# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

import torch
from mmengine import print_log
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables.base_mutable import BaseMutable


class DynamicMixin(ABC):
    """Base class for dynamic OP. A dynamic OP usually consists of a normal
    static OP and mutables, where mutables are used to control the searchable
    (mutable) part of the dynamic OP.

    Note:
        When the dynamic OP has just been initialized, its forward propagation
        logic should be the same as the corresponding static OP. Only after
        the searchable part accepts the specific mutable through the
        corresponding interface does the part really become dynamic.

    Note:
        All subclass should implement ``to_static_op`` and
        ``static_op_factory`` APIs.

    Args:
        accepted_mutables (set): The string set of all accepted mutables.
    """
    accepted_mutable_attrs: Set[str] = set()
    attr_mappings: Dict[str, str] = dict()

    @abstractmethod
    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        pass

    def get_mutable_attr(self, attr: str) -> BaseMutable:

        self.check_mutable_attr_valid(attr)
        if attr in self.attr_mappings:
            attr_map = self.attr_mappings[attr]
            return getattr(self.mutable_attrs, attr_map, None)  # type:ignore
        else:
            return getattr(self.mutable_attrs, attr, None)  # type:ignore

    @classmethod
    @abstractmethod
    def convert_from(cls, module):
        """Convert an instance of Pytorch module to a new instance of Dynamic
        module."""

    @property
    @abstractmethod
    def static_op_factory(self):
        """Corresponding Pytorch OP."""

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
        from mmrazor.models.mutables import (DerivedMutable,
                                             MutableChannelContainer)

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable `{mutable.alias}` is not fixed.')

        for mutable in self.mutable_attrs.values():  # type: ignore
            if isinstance(mutable, (MutableChannelContainer, DerivedMutable)):
                continue
            check_fixed(mutable)

    def check_mutable_attr_valid(self, attr):
        assert attr in self.attr_mappings or \
                    attr in self.accepted_mutable_attrs

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


class DynamicChannelMixin(DynamicMixin):
    """Base class for dynamic OP with mutable channels.

    Note:
        All subclass should implement ``mutable_in_channels`` and
        ``mutable_out_channels`` APIs.
    """

    attr_mappings: Dict[str, str] = {
        'in_channels': 'in_channels',
        'out_channels': 'out_channels',
    }

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


class DynamicBatchNormMixin(DynamicChannelMixin):
    """A mixin class for Pytorch BatchNorm, which can mutate
    ``num_features``."""
    accepted_mutable_attrs: Set[str] = {'num_features'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'num_features',
        'out_channels': 'num_features',
    }

    def register_mutable_attr(self, attr, mutable):
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

    def _register_mutable_attr(self, attr, mutable):

        if attr == 'num_features':
            self._register_mutable_num_features(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_num_features(
            self: _BatchNorm, mutable_num_features: BaseMutable) -> None:
        """Mutate ``num_features`` with given mutable.

        Args:
            mutable_num_features (BaseMutable): Mutable for controlling
                ``num_features``.

        Raises:
            RuntimeError: Error if both ``affine`` and
                ``tracking_running_stats`` are False.
            ValueError: Error if size of mask if not same as ``num_features``.
        """
        if not self.affine and not self.track_running_stats:
            raise RuntimeError(
                'num_features can not be mutated if both `affine` and '
                '`tracking_running_stats` are False')

        self.check_mutable_channels(mutable_num_features)
        mask_size = mutable_num_features.current_mask.size(0)
        if mask_size != self.num_features:
            raise ValueError(
                f'Expect mask size of mutable to be {self.num_features} as '
                f'`num_features`, but got: {mask_size}.')

        self.mutable_attrs['num_features'] = mutable_num_features

    def _get_num_features_mask(self: _BatchNorm) -> Optional[torch.Tensor]:
        """Get mask of ``num_features``"""
        if self.affine:
            refer_tensor = self.weight
        elif self.track_running_stats:
            refer_tensor = self.running_mean
        else:
            return None

        if 'num_features' in self.mutable_attrs:
            out_mask = self.mutable_attrs['num_features'].current_mask.to(
                refer_tensor.device)
        else:
            out_mask = torch.ones_like(refer_tensor).bool()

        return out_mask

    def get_dynamic_params(
        self: _BatchNorm
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
               Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
                Optional[Tensor]]: Sliced running_mean, running_var, weight and
                bias.
        """
        out_mask = self._get_num_features_mask()

        if self.affine:
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            running_mean = self.running_mean[out_mask] \
                if not self.training or self.track_running_stats else None
            running_var = self.running_var[out_mask] \
                if not self.training or self.track_running_stats else None
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return running_mean, running_var, weight, bias

    def to_static_op(self: _BatchNorm) -> nn.Module:
        """Convert dynamic BatchNormxd to :obj:`torch.nn.BatchNormxd`.

        Returns:
            torch.nn.BatchNormxd: :obj:`torch.nn.BatchNormxd` with sliced
                parameters.
        """
        self.check_if_mutables_fixed()

        running_mean, running_var, weight, bias = self.get_dynamic_params()
        if 'num_features' in self.mutable_attrs:
            num_features = self.mutable_attrs['num_features'].current_mask.sum(
            ).item()
        else:
            num_features = self.num_features

        static_bn = self.static_op_factory(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats)

        if running_mean is not None:
            static_bn.running_mean.copy_(running_mean)
            static_bn.running_mean = static_bn.running_mean.to(
                running_mean.device)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
            static_bn.running_var = static_bn.running_var.to(
                running_var.device)
        if weight is not None:
            static_bn.weight = nn.Parameter(weight)
        if bias is not None:
            static_bn.bias = nn.Parameter(bias)

        return static_bn


class DynamicLinearMixin(DynamicChannelMixin):
    """A mixin class for Pytorch Linear, which can mutate ``in_features`` and
    ``out_features``."""

    accepted_mutable_attrs: Set[str] = {'in_features', 'out_features'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'in_features',
        'out_channels': 'out_features',
    }

    def register_mutable_attr(self, attr, mutable):
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

    def _register_mutable_attr(self, attr, mutable):

        if attr == 'in_features':
            self._register_mutable_in_features(mutable)
        elif attr == 'out_features':
            self._register_mutable_out_features(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_in_features(
            self: nn.Linear, mutable_in_features: BaseMutable) -> None:
        """Mutate ``in_features`` with given mutable.

        Args:
            mutable_in_features (BaseMutable): Mutable for controlling
                ``in_features``.

        Raises:
            ValueError: Error if size of mask if not same as ``in_features``.
        """
        self.check_mutable_channels(mutable_in_features)
        mask_size = mutable_in_features.current_mask.size(0)
        if mask_size != self.in_features:
            raise ValueError(
                f'Expect mask size of mutable to be {self.in_features} as '
                f'`in_features`, but got: {mask_size}.')

        self.mutable_attrs['in_features'] = mutable_in_features

    def _register_mutable_out_features(
            self: nn.Linear, mutable_out_features: BaseMutable) -> None:
        """Mutate ``out_features`` with given mutable.

        Args:
            mutable_out_features (BaseMutable): Mutable for controlling
                ``out_features``.

        Raises:
            ValueError: Error if size of mask if not same as ``out_features``.
        """
        self.check_mutable_channels(mutable_out_features)
        mask_size = mutable_out_features.current_mask.size(0)
        if mask_size != self.out_features:
            raise ValueError(
                f'Expect mask size of mutable to be {self.out_features} as '
                f'`in_features`, but got: {mask_size}.')

        self.mutable_attrs['out_features'] = mutable_out_features

    def get_dynamic_params(self: nn.Linear) -> Tuple[Tensor, Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Sliced weight and bias.
        """
        if 'in_features' not in self.mutable_attrs and \
                'out_features' not in self.mutable_attrs:
            return self.weight, self.bias

        if 'in_features' in self.mutable_attrs:
            in_mask = self.mutable_attrs['in_features'].current_mask.to(
                self.weight.device)
        else:
            in_mask = torch.ones(self.weight.size(1)).bool().to(
                self.weight.device)
        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_attrs['out_features'].current_mask.to(
                self.weight.device)
        else:
            out_mask = torch.ones(self.weight.size(0)).bool().to(
                self.weight.device)

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def to_static_op(self: nn.Linear) -> nn.Module:
        """Convert to :obj:`torch.nn.Linear`.

        Returns:
            nn.Linear: :obj:`torch.nn.Linear` with sliced parameters.
        """
        self.check_if_mutables_fixed()

        weight, bias = self.get_dynamic_params()
        out_features = weight.size(0)
        in_features = weight.size(1)

        static_linear = self.static_op_factory(
            in_features=in_features,
            out_features=out_features,
            bias=True if bias is not None else False)

        static_linear.weight = nn.Parameter(weight)
        if bias is not None:
            static_linear.bias = nn.Parameter(bias)

        return static_linear


class DynamicResizeMixin(DynamicMixin):
    """A mixin class for Pytorch InputResizer, which can mutate ``shape``."""

    accepted_mutable_attrs: Set[str] = {'shape'}

    def register_mutable_attr(self, attr, mutable):
        if attr == 'shape':
            self._register_mutable_shape(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_shape(self, mutable_shape):
        assert hasattr(self, 'mutable_attrs')
        current_shape = mutable_shape.current_choice
        shape_dim = 1 if isinstance(current_shape, int) else len(current_shape)
        if shape_dim not in [1, 2, 3]:
            raise ValueError('Expect shape of mutable to be 1, 2 or 3'
                             f', but got: {shape_dim}.')

        self.mutable_attrs['shape'] = mutable_shape

    def get_dynamic_shape(self):
        if 'shape' in self.mutable_attrs:
            current_shape = self.mutable_attrs['shape'].current_choice
        else:
            current_shape = None
        return current_shape

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        input_resizer = self.static_op_factory(
            interpolation_type=self._interpolation_type,  # type:ignore
            align_corners=self._align_corners,  # type:ignore
            scale_factor=self._scale_factor)  # type:ignore

        size = self.get_dynamic_shape()
        if size is not None:
            input_resizer._size = size

        return input_resizer
