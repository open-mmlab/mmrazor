# Copyright (c) OpenMMLab. All rights reserved.

import operator
from abc import abstractmethod
from typing import Union

import torch
import torch.nn as nn
from mmengine import MMLogger

from .channel_modules import BaseChannel, BaseChannelUnit, ChannelTensor
from .module_graph import ModuleNode


class ChannelNode(ModuleNode):
    """A ChannelNode is like a torch module. It accepts  a ChannelTensor and
    output a ChannelTensor. The difference is that the torch module transforms
    a tensor, while the ChannelNode records the information of channel
    dependency in the ChannelTensor.

    Args:
        name (str): The name of the node.
        val (Union[nn.Module, str]): value of the node.
        expand_ratio (int, optional): expand_ratio compare with channel
            mask. Defaults to 1.
        module_name (str, optional): the module name of the module of the
            node.
    """

    # init

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 expand_ratio: int = 1,
                 module_name='') -> None:

        super().__init__(name, val, expand_ratio, module_name)
        self.in_channel_tensor = ChannelTensor(self.in_channels)
        self.out_channel_tensor = ChannelTensor(self.out_channels)

    @classmethod
    def copy_from(cls, node):
        """Copy from a ModuleNode."""
        assert isinstance(node, ModuleNode)
        return cls(node.name, node.val, node.expand_ratio, node.module_name)

    def reset_channel_tensors(self):
        """Reset the owning ChannelTensors."""
        self.in_channel_tensor = ChannelTensor(self.in_channels)
        self.out_channel_tensor = ChannelTensor(self.out_channels)

    # forward

    def forward(self, in_channel_tensor=None):
        """Forward with ChannelTensors."""
        assert self.in_channel_tensor is not None and \
            self.out_channel_tensor is not None
        if in_channel_tensor is None:
            out_channel_tensors = [
                node.out_channel_tensor for node in self.prev_nodes
            ]

            in_channel_tensor = out_channel_tensors
        self.channel_forward(*in_channel_tensor)
        if self.expand_ratio > 1:
            self.out_channel_tensor = self.out_channel_tensor.expand(
                self.expand_ratio)

    @abstractmethod
    def channel_forward(self, *channel_tensors: ChannelTensor):
        """Forward with ChannelTensors."""
        assert len(channel_tensors) == 1, f'{len(channel_tensors)}'
        BaseChannelUnit.union_two_units(
            list(self.in_channel_tensor.unit_dict.values())[0],
            list(channel_tensors[0].unit_dict.values())[0])

        if self.in_channels == self.out_channels:
            BaseChannelUnit.union_two_units(
                self.in_channel_tensor.unit_list[0],
                self.out_channel_tensor.unit_list[0])

    # register unit

    def register_channel_to_units(self):
        """Register the module of this node to corresponding units."""
        name = self.module_name if isinstance(self.val,
                                              nn.Module) else self.name
        for index, unit in self.in_channel_tensor.unit_dict.items():
            channel = BaseChannel(name, self.val, index, None, False,
                                  self.expand_ratio)
            if channel not in unit.input_related:
                unit.input_related.append(channel)
        for index, unit in self.out_channel_tensor.unit_dict.items():
            channel = BaseChannel(name, self.val, index, None, True,
                                  self.expand_ratio)
            if channel not in unit.output_related:
                unit.output_related.append(channel)

    # channels

    # @abstractmethod
    @property
    def in_channels(self) -> int:
        """Get the number of input channels of the node."""
        raise NotImplementedError()

    # @abstractmethod
    @property
    def out_channels(self) -> int:
        """Get the number of output channels of the node."""
        raise NotImplementedError()


# basic nodes


class PassChannelNode(ChannelNode):
    """A PassChannelNode has the same number of input channels and output
    channels.

    Besides, the corresponding input channels and output channels belong to one
    channel unit. Such as  BatchNorm, Relu.
    """

    def channel_forward(self, *in_channel_tensor: ChannelTensor):
        """Channel forward."""
        PassChannelNode._channel_forward(self, *in_channel_tensor)

    @property
    def in_channels(self) -> int:
        """Get the number of input channels of the node."""
        if len(self.prev_nodes) > 0:
            return self.prev_nodes[0].out_channels
        else:
            return 0

    @property
    def out_channels(self) -> int:
        """Get the number of output channels of the node."""
        return self.in_channels

    def __repr__(self) -> str:
        return super().__repr__() + '_pass'

    @staticmethod
    def _channel_forward(node: ChannelNode, *in_channel_tensor: ChannelTensor):
        """Channel forward."""
        assert len(in_channel_tensor) == 1 and \
            node.in_channels == node.out_channels
        in_channel_tensor[0].union(node.in_channel_tensor)
        node.in_channel_tensor.union(node.out_channel_tensor)


class MixChannelNode(ChannelNode):
    """A MixChannelNode  has independent input channels and output channels."""

    def channel_forward(self, *in_channel_tensor: ChannelTensor):
        """Channel forward."""
        assert len(in_channel_tensor) <= 1
        if len(in_channel_tensor) == 1:
            in_channel_tensor[0].union(self.in_channel_tensor)

    @property
    def in_channels(self) -> int:
        """Get the number of input channels of the node."""
        if len(self.prev_nodes) > 0:
            return self.prev_nodes[0].in_channels
        else:
            return 0

    @property
    def out_channels(self) -> int:
        """Get the number of output channels of the node."""
        if len(self.next_nodes) > 0:
            return self.next_nodes[0].in_channels
        else:
            return 0

    def __repr__(self) -> str:
        return super().__repr__() + '_mix'


class BindChannelNode(PassChannelNode):
    """A BindChannelNode has multiple inputs, and all input channels belong to
    the same channel unit."""

    def channel_forward(self, *in_channel_tensor: ChannelTensor):
        """Channel forward."""
        assert len(in_channel_tensor) > 1
        #  align channel_tensors
        ChannelTensor.align_tensors(*in_channel_tensor)

        # union tensors
        node_units = [
            channel_lis.unit_dict for channel_lis in in_channel_tensor
        ]
        for key in node_units[0]:
            BaseChannelUnit.union_units([units[key] for units in node_units])
        super().channel_forward(in_channel_tensor[0])

    def __repr__(self) -> str:
        return super(ChannelNode, self).__repr__() + '_bind'


class CatChannelNode(ChannelNode):
    """A CatChannelNode cat all input channels."""

    def channel_forward(self, *in_channel_tensors: ChannelTensor):
        BaseChannelUnit.union_two_units(self.in_channel_tensor.unit_list[0],
                                        self.out_channel_tensor.unit_list[0])
        num_ch = []
        for in_ch_tensor in in_channel_tensors:
            for start, end in in_ch_tensor.unit_dict:
                num_ch.append(end - start)

        split_units = BaseChannelUnit.split_unit(
            self.in_channel_tensor.unit_list[0], num_ch)

        i = 0
        for in_ch_tensor in in_channel_tensors:
            for in_unit in in_ch_tensor.unit_dict.values():
                BaseChannelUnit.union_two_units(split_units[i], in_unit)
                i += 1

    @property
    def in_channels(self) -> int:
        """Get the number of input channels of the node."""
        return sum([node.out_channels for node in self.prev_nodes])

    @property
    def out_channels(self) -> int:
        """Get the number of output channels of the node."""
        return self.in_channels

    def __repr__(self) -> str:
        return super().__repr__() + '_cat'


# module nodes


class ConvNode(MixChannelNode):
    """A ConvNode corresponds to a Conv2d module.

    It can deal with normal conv, dwconv and gwconv.
    """

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 expand_ratio: int = 1,
                 module_name='') -> None:
        super().__init__(name, val, expand_ratio, module_name)
        assert isinstance(self.val, nn.Conv2d)
        if self.val.groups == 1:
            self.conv_type = 'conv'
        elif self.val.in_channels == self.out_channels == self.val.groups:
            self.conv_type = 'dwconv'
        else:
            self.conv_type = 'gwconv'

    def channel_forward(self, *in_channel_tensor: ChannelTensor):
        if self.conv_type == 'conv':
            return super().channel_forward(*in_channel_tensor)
        elif self.conv_type == 'dwconv':
            return PassChannelNode._channel_forward(self, *in_channel_tensor)
        elif self.conv_type == 'gwconv':
            return super().channel_forward(*in_channel_tensor)
        else:
            pass

    @property
    def in_channels(self) -> int:
        return self.val.in_channels

    @property
    def out_channels(self) -> int:
        return self.val.out_channels

    def __repr__(self) -> str:
        return super().__repr__() + '_conv'


class LinearNode(MixChannelNode):
    """A LinearNode corresponds to a Linear module."""

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 expand_ratio: int = 1,
                 module_name='') -> None:
        super().__init__(name, val, expand_ratio, module_name)
        assert isinstance(self.val, nn.Linear)

    @property
    def in_channels(self) -> int:
        return self.val.in_features

    @property
    def out_channels(self) -> int:
        return self.val.out_features

    def __repr__(self) -> str:
        return super().__repr__() + 'linear'


class NormNode(PassChannelNode):
    """A NormNode corresponds to a BatchNorm2d module."""

    def __init__(self,
                 name: str,
                 val: Union[nn.Module, str],
                 expand_ratio: int = 1,
                 module_name='') -> None:
        super().__init__(name, val, expand_ratio, module_name)
        assert isinstance(self.val, nn.BatchNorm2d)

    @property
    def in_channels(self) -> int:
        return self.val.num_features

    @property
    def out_channels(self) -> int:
        return self.val.num_features

    def __repr__(self) -> str:
        return super().__repr__() + '_bn'


# converter


def default_channel_node_converter(node: ModuleNode) -> ChannelNode:
    """The default node converter for ChannelNode."""

    def warn(default='PassChannelNode'):
        logger = MMLogger('mmrazor', 'mmrazor')
        logger.warn((f"{node.name}({node.val}) node can't find match type of"
                     'channel_nodes,'
                     f'replaced with {default} by default.'))

    module_mapping = {
        nn.Conv2d: ConvNode,
        nn.BatchNorm2d: NormNode,
        nn.Linear: LinearNode,
    }
    function_mapping = {
        torch.add: BindChannelNode,
        torch.cat: CatChannelNode,
        operator.add: BindChannelNode
    }
    name_mapping = {
        'bind_placeholder': BindChannelNode,
        'pass_placeholder': PassChannelNode,
        'cat_placeholder': CatChannelNode,
    }
    if isinstance(node.val, nn.Module):
        # module_mapping
        for module_type in module_mapping:
            if isinstance(node.val, module_type):
                return module_mapping[module_type].copy_from(node)

    elif isinstance(node.val, str):
        for module_type in name_mapping:
            if node.val == module_type:
                return name_mapping[module_type].copy_from(node)

    else:
        for fun_type in function_mapping:
            if node.val == fun_type:
                return function_mapping[fun_type].copy_from(node)
    if len(node.prev_nodes) > 1:
        warn('BindChannelNode')
        return BindChannelNode.copy_from(node)
    else:
        warn('PassChannelNode')
        return PassChannelNode.copy_from(node)
