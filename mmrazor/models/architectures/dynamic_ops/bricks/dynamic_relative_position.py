# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Set

import torch
from mmengine import print_log
from torch import Tensor, nn

from mmrazor.models.architectures.ops import RelativePosition2D
from mmrazor.models.mutables.base_mutable import BaseMutable
from ..mixins import DynamicChannelMixin


class DynamicRelativePosition2D(RelativePosition2D, DynamicChannelMixin):
    """Searchable RelativePosition module.

    Note:
        Arguments for ``__init__`` of ``DynamicRelativePosition2D`` is totally
        same as :obj:`mmrazor.models.architectures.RelativePosition2D`.
    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `head_dims`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """

    mutable_attrs: nn.ModuleDict
    head_dims: int
    max_relative_position: int
    embeddings_table_v: nn.Parameter
    embeddings_table_h: nn.Parameter
    accepted_mutable_attrs: Set[str] = {'head_dims'}
    attr_mappings: Dict[str, str] = {
        'in_channels': 'head_dims',
        'out_channels': 'head_dims',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

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

    def to_static_op(self) -> nn.Module:
        """Convert dynamic RelativePosition2D to static One."""
        self.check_if_mutables_fixed()
        assert self.mutable_head_dims is not None

        self.current_head_dim = self.mutable_head_dims.activated_channels
        static_relative_position = self.static_op_factory(
            self.current_head_dim)
        static_relative_position.embeddings_table_v = \
            nn.Parameter(
                self.embeddings_table_v[:, :self.current_head_dim].clone())
        static_relative_position.embeddings_table_h = \
            nn.Parameter(
                self.embeddings_table_h[:, :self.current_head_dim].clone())

        return static_relative_position

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return RelativePosition2D

    @classmethod
    def convert_from(cls, module):
        """Convert a RP to a dynamic RP."""
        dynamic_rp = cls(
            head_dims=module.head_dims,
            max_relative_position=module.max_relative_position)
        return dynamic_rp

    def forward(self, length_q, length_k) -> Tensor:
        """Forward of Dynamic Relative Position."""
        if self.mutable_head_dims is None:
            self.current_head_dim = self.head_dims
        else:
            self.current_head_dim = self.mutable_head_dims.activated_channels

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
