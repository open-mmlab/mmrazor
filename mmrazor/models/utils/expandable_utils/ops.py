# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.mutables import MutableChannelContainer
from mmrazor.models.utils import get_module_device


class ExpandableMixin:
    """This minin coroperates with dynamic ops.

    It defines interfaces to expand the channels of ops. We can get a wider
    network than original supernet with it.
    """

    def expand(self, zero=False):
        """Expand the op.

        Args:
            zero (bool, optional): whether to set new weights to zero. Defaults
                to False.
        """
        return self.get_expand_op(
            self.expanded_in_channel,
            self.expanded_out_channel,
            zero=zero,
        )

    def get_expand_op(self, in_c, out_c, zero=False):
        """Get an expanded op.

        Args:
            in_c (int): New input channels
            out_c (int): New output channels
            zero (bool, optional): Whether to zero new weights. Defaults to
                False.
        """
        pass

    @property
    def _original_in_channel(self):
        """Return original in channel."""
        raise NotImplementedError()

    @property
    def _original_out_channel(self):
        """Return original out channel."""

    @property
    def expanded_in_channel(self):
        """Return expanded in channel number."""
        if self.in_mutable is not None:
            return self.in_mutable.current_mask.numel()
        else:
            return self._original_in_channel

    @property
    def expanded_out_channel(self):
        """Return expanded out channel number."""
        if self.out_mutable is not None:
            return self.out_mutable.current_mask.numel()
        else:
            return self._original_out_channel

    @property
    def mutable_in_mask(self):
        """Return the mutable in mask."""
        device = get_module_device(self)
        if self.in_mutable is not None:
            return self.in_mutable.current_mask.to(device)
        else:
            return torch.ones([self.expanded_in_channel]).to(device)

    @property
    def mutable_out_mask(self):
        """Return the mutable out mask."""
        device = get_module_device(self)
        if self.out_mutable is not None:
            return self.out_mutable.current_mask.to(device)
        else:
            return torch.ones([self.expanded_out_channel]).to(device)

    @property
    def in_mutable(self) -> MutableChannelContainer:
        """In channel mask."""
        return self.get_mutable_attr('in_channels')  # type: ignore

    @property
    def out_mutable(self) -> MutableChannelContainer:
        """Out channel mask."""
        return self.get_mutable_attr('out_channels')  # type: ignore

    def zero_weight_(self: nn.Module):
        """Zero all weights."""
        for p in self.parameters():
            p.data.zero_()

    @torch.no_grad()
    def expand_matrix(self, weight: torch.Tensor, old_weight: torch.Tensor):
        """Expand weight matrix."""
        assert len(weight.shape) == 3  # out in c
        assert len(old_weight.shape) == 3  # out in c
        mask = self.mutable_out_mask.float().unsqueeze(
            -1) * self.mutable_in_mask.float().unsqueeze(0)
        mask = mask.unsqueeze(-1).expand(*weight.shape)
        weight.data.masked_scatter_(mask.bool(), old_weight)
        return weight

    @torch.no_grad()
    def expand_vector(self, weight: torch.Tensor, old_weight: torch.Tensor):
        """Expand weight vector which has the shape of [out, c]."""
        assert len(weight.shape) == 2  # out c
        assert len(old_weight.shape) == 2  # out c
        mask = self.mutable_out_mask
        mask = mask.unsqueeze(-1).expand(*weight.shape)
        weight.data.masked_scatter_(mask.bool(), old_weight)
        return weight

    @torch.no_grad()
    def expand_bias(self, bias: torch.Tensor, old_bias: torch.Tensor):
        """Expand bias."""
        assert len(bias.shape) == 1  # out c
        assert len(old_bias.shape) == 1  # out c
        return self.expand_vector(bias.unsqueeze(-1),
                                  old_bias.unsqueeze(-1)).squeeze(1)


class ExpandableConv2d(dynamic_ops.DynamicConv2d, ExpandableMixin):

    @property
    def _original_in_channel(self):
        return self.in_channels

    @property
    def _original_out_channel(self):
        return self.out_channels

    def get_expand_op(self, in_c, out_c, zero=False):

        if self.groups == 1:
            return self._get_expand_op_normal_conv(in_c, out_c, zero=zero)
        elif self.in_channels == self.out_channels == self.groups:
            return self._get_expand_op_dw_conv(in_c, out_c, zero=zero)
        else:
            raise NotImplementedError('Groupwise conv is not supported yet.')

    def _get_expand_op_normal_conv(self, in_c, out_c, zero=False):

        module = nn.Conv2d(in_c, out_c, self.kernel_size, self.stride,
                           self.padding, self.dilation, self.groups, self.bias
                           is not None,
                           self.padding_mode).to(get_module_device(self))
        if zero:
            ExpandableMixin.zero_weight_(module)

        weight = self.expand_matrix(
            module.weight.flatten(2), self.weight.flatten(2))
        module.weight.data = weight.reshape(module.weight.shape)
        if module.bias is not None and self.bias is not None:
            bias = self.expand_vector(
                module.bias.unsqueeze(-1), self.bias.unsqueeze(-1))
            module.bias.data = bias.reshape(module.bias.shape)
        return module

    def _get_expand_op_dw_conv(self, in_c, out_c, zero=False):
        assert in_c == out_c
        module = nn.Conv2d(in_c, out_c, self.kernel_size, self.stride,
                           self.padding, self.dilation, in_c, self.bias
                           is not None,
                           self.padding_mode).to(get_module_device(self))
        if zero:
            ExpandableMixin.zero_weight_(module)

        weight = self.expand_vector(
            module.weight.flatten(1), self.weight.flatten(1))
        module.weight.data = weight.reshape(module.weight.shape)
        if module.bias is not None and self.bias is not None:
            bias = self.expand_vector(
                module.bias.unsqueeze(-1), self.bias.unsqueeze(-1))
            module.bias.data = bias.reshape(module.bias.shape)
        return module


class ExpandLinear(dynamic_ops.DynamicLinear, ExpandableMixin):

    @property
    def _original_in_channel(self):
        return self.in_features

    @property
    def _original_out_channel(self):
        return self.out_features

    def get_expand_op(self, in_c, out_c, zero=False):
        module = nn.Linear(in_c, out_c, self.bias
                           is not None).to(get_module_device(self))
        if zero:
            ExpandableMixin.zero_weight_(module)

        weight = self.expand_matrix(
            module.weight.unsqueeze(-1), self.weight.unsqueeze(-1))
        module.weight.data = weight.reshape(module.weight.shape)
        if module.bias is not None:
            bias = self.expand_vector(
                module.bias.unsqueeze(-1), self.bias.unsqueeze(-1))
            module.bias.data = bias.reshape(module.bias.shape)
        return module


class ExpandableBatchNorm2d(dynamic_ops.DynamicBatchNorm2d, ExpandableMixin):

    @property
    def _original_in_channel(self):
        return self.num_features

    @property
    def _original_out_channel(self):
        return self.num_features

    def get_expand_op(self, in_c, out_c, zero=False):
        assert in_c == out_c
        module = nn.BatchNorm2d(in_c, self.eps, self.momentum, self.affine,
                                self.track_running_stats).to(
                                    get_module_device(self))
        if zero:
            ExpandableMixin.zero_weight_(module)

        if module.running_mean is not None:
            module.running_mean.data = self.expand_bias(
                module.running_mean, self.running_mean)

        if module.running_var is not None:
            module.running_var.data = self.expand_bias(module.running_var,
                                                       self.running_var)
        module.weight.data = self.expand_bias(module.weight, self.weight)
        module.bias.data = self.expand_bias(module.bias, self.bias)
        return module
