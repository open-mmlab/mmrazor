# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Optional, Set, Tuple

import torch
from mmengine import print_log
from torch import Tensor, nn
from torch.nn import LayerNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_mixins import DynamicChannelMixin


class DynamicLayerNormMixin(DynamicChannelMixin):
    """A mixin class for Pytorch LayerNorm, which can mutate
    ``num_features``."""
    accepted_mutable_attrs: Set[str] = {'num_features'}

    attr_mappings: Dict[str, str] = {
        'in_channels': 'num_features',
        'out_channels': 'num_features',
    }

    @property
    def num_features(self):
        return getattr(self, 'normalized_shape')[0]

    @property
    def mutable_num_features(self):
        """Mutable number of features."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['num_features']

    def register_mutable_attr(self, attr, mutable):
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

    def _register_mutable_attr(self, attr, mutable):
        """Register `num_features`."""
        if attr == 'num_features':
            self._register_mutable_num_features(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_num_features(
            self: LayerNorm, mutable_num_features: BaseMutable) -> None:
        """Mutate ``num_features`` with given mutable.

        Args:
            mutable_num_features (BaseMutable): Mutable for controlling
                ``num_features``.
        Raises:
            RuntimeError: Error if both ``affine`` and
                ``tracking_running_stats`` are False.
            ValueError: Error if size of mask if not same as ``num_features``.
        """
        if not self.elementwise_affine:
            raise RuntimeError(
                'num_features can not be mutated if both `affine` and '
                '`tracking_running_stats` are False')

        self.check_mutable_channels(mutable_num_features)
        mask_size = mutable_num_features.current_mask.size(0)

        # normalized_shape is a tuple
        if mask_size != self.normalized_shape[0]:
            raise ValueError(
                f'Expect mask size of mutable to be {self.normalized_shape}'
                f' as `normalized_shape`, but got: {mask_size}.')

        self.mutable_attrs['num_features'] = mutable_num_features

    def _get_num_features_mask(self: LayerNorm) -> Optional[torch.Tensor]:
        """Get mask of ``num_features``."""
        if self.elementwise_affine:
            refer_tensor = self.weight
        else:
            return None

        if 'num_features' in self.mutable_attrs:
            out_mask = self.mutable_num_features.current_mask.to(
                refer_tensor.device)
        else:
            out_mask = torch.ones_like(refer_tensor).bool()

        return out_mask

    def get_dynamic_params(
            self: LayerNorm) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
                Optional[Tensor]]: Sliced running_mean, running_var, weight and
                bias.
        """
        out_mask = self._get_num_features_mask()

        if self.elementwise_affine:
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        return weight, bias

    def to_static_op(self: LayerNorm) -> nn.Module:
        """Convert dynamic LayerNormxd to :obj:`torch.nn.LayerNormxd`.

        Returns:
            torch.nn.LayerNormxd: :obj:`torch.nn.LayerNormxd` with sliced
                parameters.
        """
        self.check_if_mutables_fixed()

        weight, bias = self.get_dynamic_params()

        if 'num_features' in self.mutable_attrs:
            num_features = self.mutable_attrs['num_features'].current_mask.sum(
            ).item()
        else:
            num_features = self.num_features

        static_ln = self.static_op_factory(
            normalized_shape=num_features,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine)

        if weight is not None:
            static_ln.weight = nn.Parameter(weight.clone())
        if bias is not None:
            static_ln.bias = nn.Parameter(bias.clone())

        return static_ln
