# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

import torch
from mmcls.models.utils import PatchEmbed
from mmengine import print_log
from torch import Tensor, nn
from torch.nn import LayerNorm, Linear, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.ops import (MultiheadAttention,
                                              RelativePosition2D)
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.mutables.mutable_value import MutableValue
from .dynamic_protocol import DynamicMHAProtocol, DynamicRPProtocol


class DynamicMixin(ABC):
    """Base class for dynamic OP.

    A dynamic OP usually consists of a normal
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
        """Register attribute of mutable."""
        pass

    def get_mutable_attr(self, attr: str) -> BaseMutable:
        """Access the mutable attributes."""
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

        def check_fixed(mutable: Optional[BaseMutable]) -> None:
            if mutable is not None and not mutable.is_fixed:
                raise RuntimeError(f'Mutable {type(mutable)} is not fixed.')

        for mutable in self.mutable_attrs.values():  # type: ignore
            check_fixed(mutable)

    def check_mutable_attr_valid(self, attr):
        """Check whether the attribute is valid."""
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
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
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

    @property
    def mutable_in_features(self: Linear) -> nn.Module:
        """Mutable input feature dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['in_features']

    @property
    def mutable_out_features(self: Linear) -> nn.Module:
        """Mutable output feature dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['out_features']

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
        """Register attr `in_features` and `out_features`."""
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
            in_mask = self.mutable_in_features.current_mask.to(
                self.weight.device)
        else:
            in_mask = torch.ones(self.weight.size(1)).bool().to(
                self.weight.device)
        if 'out_features' in self.mutable_attrs:

            out_mask = self.mutable_out_features.current_mask.to(
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

        static_linear.weight = nn.Parameter(weight.clone())
        if bias is not None:
            static_linear.bias = nn.Parameter(bias.clone())

        return static_linear


class DynamicPatchEmbedMixin(DynamicChannelMixin):

    accepted_mutable_attrs: Set[str] = {'embed_dims'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'embed_dims',
        'out_channels': 'embed_dims'
    }

    @property
    def mutable_embed_dims(self):
        """Mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['embed_dims']

    def register_mutable_attr(self: PatchEmbed, attr: str,
                              mutable: BaseMutable):
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
        """Register `embed_dims`."""
        if attr == 'embed_dims':
            self._register_embed_dims(mutable)
        else:
            raise NotImplementedError

    def _register_embed_dims(self: PatchEmbed,
                             mutable_patch_embedding: BaseMutable) -> None:
        """Register mutable embedding dimension."""
        mask_size = mutable_patch_embedding.current_mask.size(0)

        if mask_size != self.embed_dims:
            raise ValueError(
                f'Expect mask size of mutable to be {self.embed_dims} as '
                f'`embed_dims`, but got: {mask_size}.')

        self.mutable_attrs['embed_dims'] = mutable_patch_embedding

    def _get_dynamic_params(self: PatchEmbed) -> torch.Tensor:
        """Get mask of ``embed_dims``"""
        if 'embed_dims' not in self.mutable_attrs:
            return self.projection.weight, self.projection.bias
        else:
            out_mask = self.mutable_embed_dims.current_mask.to(
                self.projection.weight.device)
            weight = self.projection.weight[out_mask][:]
            bias = self.projection.bias[
                out_mask] if self.projection.bias is not None else None  # noqa: E501
            return weight, bias

    def to_static_op(self: PatchEmbed) -> nn.Module:
        """Convert dynamic PatchEmbed to static PatchEmbed."""
        self.check_if_mutables_fixed()
        assert self.mutable_embed_dims is not None

        weight, bias = self._get_dynamic_params()
        static_patch_embed = self.static_op_factory(
            img_size=self.img_size,
            in_channels=3,
            embed_dims=self.mutable_embed_dims.current_choice)

        static_patch_embed.projection.weight = nn.Parameter(weight.clone())
        static_patch_embed.projection.bias = nn.Parameter(bias.clone())

        return static_patch_embed


class DynamicRelativePosition2DMixin(DynamicChannelMixin, DynamicRPProtocol):

    accepted_mutable_attrs: Set[str] = {'head_dims'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'head_dims',
        'out_channels': 'head_dims',
    }

    @property
    def mutable_head_dims(self):
        """Mutable head dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['head_dims']

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

    def _register_mutable_attr(self, attr, mutable):
        """Register `head_dims`"""
        if attr == 'head_dims':
            self._registry_mutable_head_dims(mutable)
        else:
            raise NotImplementedError

    def _registry_mutable_head_dims(self,
                                    mutable_head_dims: BaseMutable) -> None:
        """Register head dimension."""
        assert hasattr(self, 'mutable_attrs')
        self.mutable_attrs['head_dims'] = mutable_head_dims

    def forward_mixin(self, length_q, length_k) -> Tensor:
        """Forward of Relative Position."""
        if self.mutable_head_dims is None:
            self.current_head_dim = self.head_dims
        else:
            self.current_head_dim = self.mutable_head_dims.current_choice

        self.sample_eb_table_h = self.embeddings_table_h[:, :self.
                                                         current_head_dim]
        self.sample_eb_table_v = self.embeddings_table_v[:, :self.
                                                         current_head_dim]

        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        # get the embeddings with the corresponding distance

        embeddings = self.sample_eb_table_v[final_mat_v] + \
            self.sample_eb_table_h[final_mat_h]

        return embeddings

    def to_static_op(self) -> nn.Module:
        """Convert dynamic RelativePosition2D to static One."""
        self.check_if_mutables_fixed()

        self.current_head_dim = self.mutable_head_dims.current_choice
        static_relative_position = RelativePosition2D(self.current_head_dim)
        static_relative_position.embeddings_table_v = \
            nn.Parameter(
                self.embeddings_table_v[:, :self.current_head_dim].clone())
        static_relative_position.embeddings_table_h = \
            nn.Parameter(
                self.embeddings_table_h[:, :self.current_head_dim].clone())

        return static_relative_position


class DynamicMHAMixin(DynamicMixin, DynamicMHAProtocol):
    """Mixins for Dynamic Multi head attention.

    Note:
        `embed_dims` serve the in_dim of qkv and out_dim of proj
        `q_embed_dims` serve the out_dim of qkv and in_dim of proj
        `q_embed_dims` is a DerivedMutable derived from `num_heads`
            with `num_heads` \times 64.
    """
    accepted_mutable_attrs: Set[str] = {
        'num_heads', 'embed_dims', 'q_embed_dims'
    }

    @property
    def mutable_num_heads(self):
        """Mutable number of heads."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['num_heads']

    @property
    def mutable_embed_dims(self):
        """Mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['embed_dims']

    @property
    def mutable_q_embed_dims(self):
        """Mutable intermediate embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['q_embed_dims']

    def register_mutable_attr(self, attr: str, mutable: BaseMutable):
        """Register attribute of mutable."""
        if attr == 'num_heads':
            self._register_mutable_num_heads(mutable)
        elif attr == 'embed_dims':
            self._register_mutable_embed_dims(mutable)
        elif attr == 'q_embed_dims':
            self._register_mutable_q_embed_dims(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_num_heads(self, mutable_num_heads):
        """Register the mutable number of heads."""
        assert hasattr(self, 'mutable_attrs')
        current_choice = mutable_num_heads.current_choice
        if current_choice > self.num_heads:
            raise ValueError(
                f'Expect value of mutable to be smaller or equal than '
                f'{self.num_heads} as `num_heads`, but got: {current_choice}.')

        self.mutable_attrs['num_heads'] = mutable_num_heads

    def _register_mutable_embed_dims(self, mutable_embed_dims):
        """Register mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        mask_size = mutable_embed_dims.current_mask.size(0)
        if mask_size != self.embed_dims:
            raise ValueError(
                f'Expect mask size of mutable to be {self.embed_dims} as '
                f'`embed_dims`, but got: {mask_size}.')

        self.mutable_attrs['embed_dims'] = mutable_embed_dims

    def _register_mutable_q_embed_dims(self, mutable_q_embed_dims):
        """Register intermediate mutable embedding dimension."""
        assert hasattr(self, 'mutable_attrs')
        self.mutable_attrs['q_embed_dims'] = mutable_q_embed_dims

    def _get_dynamic_proj_params(self, w: nn.Linear) -> Tuple[Tensor, Tensor]:
        """Get parameters of dynamic projection.

        Note:
            The input dimension is decided by `mutable_q_embed_dims`.
            The output dimension is decided by `mutable_embed_dims`.
        """
        # TODO support mask
        if self.mutable_embed_dims is None and \
                self.mutable_q_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_q_embed_dims is not None:
            in_features = self.mutable_q_embed_dims.current_choice
        else:
            in_features = self.embed_dims

        if self.mutable_embed_dims is not None:
            out_features = self.mutable_embed_dims.current_choice
        else:
            out_features = self.embed_dims

        weight = w.weight[:out_features, :in_features]
        bias = w.bias[:out_features] if w.bias is not None else None

        return weight, bias

    def _get_dynamic_qkv_params(self, w: nn.Linear) -> Tuple[Tensor, Tensor]:
        """Get parameters of dynamic QKV.

        Note:
            The output dimension is decided by `mutable_q_embed_dims`.
            The input dimension is decided by `mutable_embed_dims`.
        """
        # TODO support mask later
        if self.mutable_q_embed_dims is None and \
                self.mutable_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_embed_dims is not None:
            in_features = self.mutable_embed_dims.current_choice
        else:
            in_features = self.embed_dims

        if self.mutable_q_embed_dims is not None:
            out_features = self.mutable_q_embed_dims.current_choice
        else:
            out_features = self.embed_dims

        weight = w.weight[:out_features, :in_features]
        bias = w.bias[:out_features] if w.bias is not None else None

        return weight, bias

    def to_static_op(self) -> nn.Module:
        """Convert dynamic MultiheadAttention to static one."""
        self.check_if_mutables_fixed()

        embed_dims = self.mutable_embed_dims.current_choice
        num_heads = self.mutable_num_heads.current_choice

        q_w, q_b = self._get_dynamic_qkv_params(self.w_qs)
        k_w, k_b = self._get_dynamic_qkv_params(self.w_ks)
        v_w, v_b = self._get_dynamic_qkv_params(self.w_vs)

        proj_w, proj_b = self._get_dynamic_proj_params(self.proj)

        static_mha = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=None,
            attn_drop_rate=self.attn_drop_rate,
            relative_position=self.relative_position,
            max_relative_position=self.max_relative_position)

        static_mha.w_qs.weight = nn.Parameter(q_w.clone())
        static_mha.w_qs.bias = nn.Parameter(q_b.clone())

        static_mha.w_ks.weight = nn.Parameter(k_w.clone())
        static_mha.w_ks.bias = nn.Parameter(k_b.clone())

        static_mha.w_vs.weight = nn.Parameter(v_w.clone())
        static_mha.w_vs.bias = nn.Parameter(v_b.clone())

        static_mha.proj.weight = nn.Parameter(proj_w.clone())
        static_mha.proj.bias = nn.Parameter(proj_b.clone())

        if self.relative_position:
            static_mha.rel_pos_embed_k = self.rel_pos_embed_k.to_static_op()
            static_mha.rel_pos_embed_v = self.rel_pos_embed_v.to_static_op()

        return static_mha


class DynamicSequentialMixin(DynamicMixin):

    accepted_mutable_attrs: Set[str] = {'depth'}

    @property
    def mutable_depth(self: Sequential) -> nn.Module:
        """Mutable depth."""
        assert hasattr(self, 'mutable_attrs')
        return self.mutable_attrs['depth']

    def register_mutable_attr(self: Sequential, attr: str,
                              mutable: BaseMutable):
        """Register attribute of mutable."""
        if attr == 'depth':
            self._register_mutable_depth(mutable)
        else:
            raise NotImplementedError

    def _register_mutable_depth(self: Sequential, mutable_depth: MutableValue):
        """Register mutable depth."""
        assert hasattr(self, 'mutable_attrs')
        assert mutable_depth.current_choice is not None
        current_depth = mutable_depth.current_choice
        if current_depth > len(self._modules):
            raise ValueError(f'Expect depth of mutable to be smaller than '
                             f'{len(self._modules)} as `depth`, '
                             f'but got: {current_depth}.')
        self.mutable_attrs['depth'] = mutable_depth

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return Sequential

    def to_static_op(self: Sequential) -> Sequential:
        """Convert dynamic Sequential to static one."""
        self.check_if_mutables_fixed()

        if self.mutable_depth is None:
            fixed_depth = len(self)
        else:
            fixed_depth = self.get_current_choice(self.mutable_depth)

        modules = []
        passed_module_nums = 0
        for module in self:
            if isinstance(module, self.forward_ignored_module):
                continue
            else:
                passed_module_nums += 1
            if passed_module_nums > fixed_depth:
                break

            modules.append(module)

        return Sequential(*modules)


class DynamicLayerNormMixin(DynamicChannelMixin):
    """A mixin class for Pytorch LayerNorm, which can mutate
    ``num_features``."""
    accepted_mutable_attrs: Set[str] = {'num_features'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'num_features',
        'out_channels': 'num_features',
    }

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
