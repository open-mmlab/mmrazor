# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm
from torch import distributed as dist

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.mutables.mutable_channel.mutable_channel_container import \
    MutableChannelContainer
from mmrazor.models.mutables.mutable_channel.units.l1_mutable_channel_unit import \
    L1MutableChannelUnit  # noqa
from mmrazor.registry import MODELS
from .ops import GroupFisherConv2d, GroupFisherLinear, GroupFisherMixin


@MODELS.register_module()
class GroupFisherChannelUnit(L1MutableChannelUnit):
    """ChannelUnit for GroupFisher Pruning Algorithm.

    Args:
        num_channels (int): Number of channels.
        normalization_type (str): Type of normalization. It can be one of
            ['flops','act','none',]. Defaults to 'flop'.
        mutate_linear (bool): Whether to prune linear layers.
    """

    def __init__(self,
                 num_channels: int,
                 normalization_type: str = 'flops',
                 mutate_linear=False,
                 *args) -> None:
        super().__init__(num_channels, *args)
        normalized_fisher_info = torch.zeros([self.num_channels])
        self.register_buffer('normalized_fisher_info', normalized_fisher_info)
        self.normalized_fisher_info: torch.Tensor

        self.hook_handles: List = []
        assert normalization_type in ['flops', 'act', 'none']
        self.delta_type = normalization_type

        self.mutate_linear = mutate_linear

    def prepare_for_pruning(self, model: nn.Module) -> None:
        """Prepare for pruning, including register mutable channels.

        Args:
            model (nn.Module): The model need to be pruned.
        """
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: GroupFisherConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: GroupFisherLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    # prune
    def try_to_prune_min_channel(self) -> bool:
        """Prune the channel with the minimum value of fisher information."""
        if self.mutable_channel.activated_channels > 1:
            imp = self.importance()
            index = imp.argmin()
            self.mutable_channel.mask.scatter_(0, index, 0.0)
            return True
        else:
            return False

    @property
    def is_mutable(self) -> bool:
        """Whether the unit is mutable."""
        mutable = super().is_mutable
        if self.mutate_linear:
            return mutable
        else:
            has_linear = False
            for layer in self.input_related:
                if isinstance(layer.module, nn.Linear):
                    has_linear = True
            return mutable and (not has_linear)

    @property
    def input_related_dynamic_ops(self):
        for channel in self.input_related:
            if isinstance(channel.module, GroupFisherMixin):
                yield channel.module

    @property
    def output_related_dynamic_ops(self):
        for channel in self.output_related:
            if isinstance(channel.module, GroupFisherMixin):
                yield channel.module

    @property
    def dynamic_ops(self):
        for module in self.input_related_dynamic_ops:
            yield module
        for module in self.output_related_dynamic_ops:
            yield module

    # fisher information recorded

    def start_record_fisher_info(self) -> None:
        """Start recording the related fisher info of each channel."""
        for module in self.dynamic_ops:
            module.start_record()

    def end_record_fisher_info(self) -> None:
        """Stop recording the related fisher info of each channel."""
        for module in self.dynamic_ops:
            module.end_record()

    def reset_recorded(self) -> None:
        """Reset the recorded info of each channel."""
        for module in self.dynamic_ops:
            module.reset_recorded()

    # fisher related computation

    def importance(self):
        """The importance of each channel."""
        fisher = self.normalized_fisher_info.clone()
        mask = self.mutable_channel.current_mask
        n_mask = (1 - mask.float()).bool()
        fisher.masked_fill_(n_mask, fisher.max() + 1)
        return fisher

    def reset_fisher_info(self) -> None:
        """Reset the related fisher info."""
        self.normalized_fisher_info.zero_()

    @torch.no_grad()
    def update_fisher_info(self) -> None:
        """Update the fisher info of each channel."""

        batch_fisher_sum = self.current_batch_fisher
        assert isinstance(batch_fisher_sum, torch.Tensor)
        if dist.is_initialized():
            dist.all_reduce(batch_fisher_sum)
        batch_fisher_sum = self._get_normalized_fisher_info(
            batch_fisher_sum, self.delta_type)
        self.normalized_fisher_info = self.normalized_fisher_info + batch_fisher_sum  # noqa

    @property
    def current_batch_fisher(self) -> torch.Tensor:
        """Accumulate the unit's fisher info of this batch."""
        with torch.no_grad():
            fisher: torch.Tensor = 0
            for module in self.input_related_dynamic_ops:
                fisher = fisher + self._fisher_of_a_module(module)
            return (fisher**2).sum(0)  # shape: [C]

    @torch.no_grad()
    def _fisher_of_a_module(self, module: GroupFisherMixin) -> torch.Tensor:
        """Calculate the fisher info of one module.

        Args:
            module (GroupFisherConv2d): A `GroupFisherConv2d` module.

        Return:
            torch.Tensor: Whose shape is [B C]
        """
        assert len(module.recorded_input) > 0 and \
            len(module.recorded_input) == len(module.recorded_grad)
        fisher_sum: torch.Tensor = 0
        for input, grad_input in zip(module.recorded_input,
                                     module.recorded_grad):
            fisher: torch.Tensor = input * grad_input
            if len(fisher.shape) == 4:
                fisher = fisher.sum(dim=[2, 3])
            assert len(fisher.shape) == 2  # B C
            fisher_sum = fisher_sum + fisher
        assert isinstance(fisher_sum, torch.Tensor)
        # expand to full num_channel
        batch_size = fisher_sum.shape[0]
        mask = self.mutable_channel.current_mask.unsqueeze(0).expand(
            [batch_size, self.num_channels])
        zeros = fisher_sum.new_zeros([batch_size, self.num_channels])
        fisher_sum = zeros.masked_scatter_(mask, fisher_sum)
        return fisher_sum

    @torch.no_grad()
    def _get_normalized_fisher_info(self,
                                    fisher_info,
                                    delta_type='flop') -> torch.Tensor:
        """Get the normalized fisher info.

        Args:
            delta_type (str): Type of delta. Defaults to 'flop'.
        """
        fisher = fisher_info.double()
        if delta_type == 'flops':
            delta_flop = self._delta_flop_of_a_channel
            assert delta_flop > 0
            fisher = fisher / (float(delta_flop) / 1e9)
        elif delta_type == 'act':
            delta_memory = self._delta_memory_of_a_channel
            assert delta_memory > 0
            fisher = fisher / (float(delta_memory) / 1e6)
        elif delta_type == 'none':
            pass
        else:
            raise NotImplementedError(delta_type)
        return fisher

    @property
    def _delta_flop_of_a_channel(self) -> torch.Tensor:
        """Calculate the flops of a channel."""
        delta_flop = 0
        for module in self.output_related_dynamic_ops:
            delta_flop += module.delta_flop_of_a_out_channel
        for module in self.input_related_dynamic_ops:
            delta_flop += module.delta_flop_of_a_in_channel
        return delta_flop

    @property
    def _delta_memory_of_a_channel(self) -> torch.Tensor:
        """Calculate the memory of a channel."""
        delta_memory = 0
        for module in self.output_related_dynamic_ops:
            delta_memory += module.delta_memory_of_a_out_channel
        return delta_memory
