# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import Tensor

from mmrazor.registry import MODELS


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise (stride=2).

    Args:
        in_channels (int): number of channels of input tensor.
        out_channels (int): number of channels of output tensor.
        act_cfg (Dict): config to build activation layer.
        norm_cfg (Dict): config to build normalization layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_cfg: Dict = dict(type='ReLU'),
        norm_cfg: Dict = dict(type='BN')
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.relu = build_activation_layer(self.act_cfg)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels // 2,
            1,
            stride=2,
            padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(
            self.in_channels,
            self.out_channels // 2,
            1,
            stride=2,
            padding=0,
            bias=False)
        self.bn = build_norm_layer(self.norm_cfg, self.out_channels)[1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward with factorized reduce."""
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class StandardConv(nn.Module):
    """Standard Convolution in Darts. Basic structure is ReLU-Conv-BN.

    Args:
        in_channels (int): number of channels of input tensor.
        out_channels (int): number of channels of output tensor.
        kernel_size (Union[int, Tuple]): size of the convolving kernel.
        stride (Union[int, Tuple]): controls the stride for the
            cross-correlation, a single number or a one-element tuple.
            Default to 1.
        padding (Union[str, int, Tuple]): Padding added to both sides
            of the input. Default to 0.
        act_cfg (Dict): config to build activation layer.
        norm_cfg (Dict): config to build normalization layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[str, int, Tuple] = 0,
        act_cfg: Dict = dict(type='ReLU'),
        norm_cfg: Dict = dict(type='BN')
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(
            build_activation_layer(self.act_cfg),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=False),
            build_norm_layer(self.norm_cfg, self.out_channels)[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward the standard convolution."""
        return self.net(x)


class Node(nn.Module):
    """Node structure of DARTS.

    Args:
        node_id (str): key of the node.
        num_prev_nodes (int): number of previous nodes.
        channels (int): number of channels of current node.
        num_downsample_nodes (int): index of downsample node.
        mutable_cfg (Dict): config of `DiffMutableModule`.
        route_cfg (Dict): config of `DiffChoiceRoute`.
    """

    def __init__(self, node_id: str, num_prev_nodes: int, channels: int,
                 num_downsample_nodes: int, mutable_cfg: Dict,
                 route_cfg: Dict) -> None:
        super().__init__()
        edges = nn.ModuleDict()
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_nodes else 1
            edge_id = f'{node_id}_p{i}'

            module_kwargs = dict(
                in_channels=channels,
                out_channels=channels,
                stride=stride,
            )

            mutable_cfg.update(module_kwargs=module_kwargs)
            mutable_cfg.update(alias=edge_id)
            edges.add_module(edge_id, MODELS.build(mutable_cfg))

        route_cfg.update(alias=node_id)
        route_cfg.update(edges=edges)
        self.route = MODELS.build(route_cfg)

    def forward(self, prev_nodes: Union[List[Tensor],
                                        Tuple[Tensor]]) -> Tensor:
        """Forward with the previous nodes list."""
        return self.route(prev_nodes)


class Cell(nn.Module):
    """Darts cell structure.

    Args:
        num_nodes (int): number of nodes.
        channels (int): number of channels of current cell.
        prev_channels (int): number of channel of previous input.
        prev_prev_channels (int): number of channel of previous previous input.
        reduction (bool): whether to reduce the feature map size.
        prev_reduction (bool): whether to reduce the previous feature map size.
        mutable_cfg (Optional[Dict]): config of `DiffMutableModule`.
        route_cfg (Optional[Dict]): config of `DiffChoiceRoute`.
        act_cfg (Dict): config to build activation layer.
            Defaults to dict(type='ReLU').
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(
            self,
            num_nodes: int,
            channels: int,
            prev_channels: int,
            prev_prev_channels: int,
            reduction: bool,
            prev_reduction: bool,
            mutable_cfg: Dict,
            route_cfg: Dict,
            act_cfg: Dict = dict(type='ReLU'),
            norm_cfg: Dict = dict(type='BN'),
    ) -> None:

        super().__init__()
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.reduction = reduction
        self.num_nodes = num_nodes

        # If previous cell is reduction cell, current input size does not match
        # with output size of cell[k-2]. So the output[k-2] should be reduced
        # by preprocessing.
        if prev_reduction:
            self.preproc0 = FactorizedReduce(prev_prev_channels, channels,
                                             self.act_cfg, self.norm_cfg)
        else:
            self.preproc0 = StandardConv(prev_prev_channels, channels, 1, 1, 0,
                                         self.act_cfg, self.norm_cfg)
        self.preproc1 = StandardConv(prev_channels, channels, 1, 1, 0,
                                     self.act_cfg, self.norm_cfg)

        # generate dag
        self.nodes = nn.ModuleList()
        for depth in range(2, self.num_nodes + 2):
            if reduction:
                node_id = f'reduce_n{depth}'
                num_downsample_nodes = 2
            else:
                node_id = f'normal_n{depth}'
                num_downsample_nodes = 0
            self.nodes.append(
                Node(node_id, depth, channels, num_downsample_nodes,
                     mutable_cfg, route_cfg))

    def forward(self, s0: Tensor, s1: Tensor) -> Tensor:
        """Forward with the outputs of previous previous cell and previous
        cell."""
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.nodes:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        return torch.cat(tensors[2:], dim=1)


class AuxiliaryModule(nn.Module):
    """Auxiliary head in 2/3 place of network to let the gradient flow well.

    Args:
        in_channels (int): number of channels of inputs.
        base_channels (int): number of middle channels of the auxiliary module.
        out_channels (int): number of channels of outputs.
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int,
                 out_channels: int,
                 norm_cfg: Dict = dict(type='BN')) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2, padding=0,
                         count_include_pad=False),  # 2x2 out
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            build_norm_layer(self.norm_cfg, base_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=2,
                      bias=False),  # 1x1 out
            build_norm_layer(self.norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """Forward the auxiliary module."""
        return self.net(x)


@MODELS.register_module()
class DartsBackbone(nn.Module):
    """Backbone of Differentiable Architecture Search (DARTS).

    Args:
        in_channels (int): number of channels of input tensor.
        base_channels (int): number of middle channels.
        mutable_cfg (Optional[Dict]): config of `DiffMutableModule`.
        route_cfg (Optional[Dict]): config of `DiffChoiceRoute`.
        num_layers (Optional[int]): number of layers.
            Defaults to 8.
        num_nodes (Optional[int]): number of nodes.
            Defaults to 4.
        stem_multiplier (Optional[int]): multiplier for stem.
            Defaults to 3.
        out_indices (tuple, optional): output indices for auxliary module.
            Defaults to (7, ).
        auxliary (bool, optional): whether use auxliary module.
            Defaults to False.
        aux_channels (Optional[int]): number of middle channels of
            auxliary module. Defaults to None.
        aux_out_channels (Optional[int]): number of output channels of
            auxliary module. Defaults to None.
        act_cfg (Dict): config to build activation layer.
            Defaults to dict(type='ReLU').
        norm_cfg (Dict): config to build normalization layer.
            Defaults to dict(type='BN').
    """

    def __init__(
            self,
            in_channels: int,
            base_channels: int,
            mutable_cfg: Dict,
            route_cfg: Dict,
            num_layers: int = 8,
            num_nodes: int = 4,
            stem_multiplier: int = 3,
            out_indices: Union[Tuple, List] = (7, ),
            auxliary: bool = False,
            aux_channels: Optional[int] = None,
            aux_out_channels: Optional[int] = None,
            act_cfg: Dict = dict(type='ReLU'),
            norm_cfg: Dict = dict(type='BN'),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.stem_multiplier = stem_multiplier
        self.out_indices = out_indices
        assert self.out_indices[-1] == self.num_layers - 1
        if auxliary:
            assert aux_channels is not None
            assert aux_out_channels is not None
            self.aux_channels = aux_channels
            self.aux_out_channels = aux_out_channels
            self.auxliary_indice = 2 * self.num_layers // 3

        else:
            self.auxliary_indice = -1
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = self.stem_multiplier * self.base_channels
        stem_norm_cfg = copy.deepcopy(self.norm_cfg)
        stem_norm_cfg.update(dict(affine=True))
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.out_channels, 3, 1, 1, bias=False),
            build_norm_layer(self.norm_cfg, self.out_channels)[1])

        # for the first cell, stem is used for both s0 and s1
        # prev_prev_channels and prev_channels is output channel size,
        # but c_cur is input channel size.
        prev_prev_channels = self.out_channels
        prev_channels = self.out_channels
        self.out_channels = self.base_channels

        self.cells = nn.ModuleList()
        prev_reduction, reduction = False, False
        for i in range(self.num_layers):
            prev_reduction, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3
            # and 2/3 layer.
            if i in [self.num_layers // 3, 2 * self.num_layers // 3]:
                self.out_channels *= 2
                reduction = True

            cell = Cell(self.num_nodes, self.out_channels, prev_channels,
                        prev_prev_channels, reduction, prev_reduction,
                        mutable_cfg, route_cfg, self.act_cfg, self.norm_cfg)
            self.cells.append(cell)

            prev_prev_channels = prev_channels
            prev_channels = self.out_channels * self.num_nodes

            if i == self.auxliary_indice:
                self.auxliary_module = AuxiliaryModule(prev_channels,
                                                       self.aux_channels,
                                                       self.aux_out_channels,
                                                       self.norm_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward the darts backbone."""
        outs = []
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i in self.out_indices:
                outs.append(s1)
            if i == self.auxliary_indice and self.training:
                aux_feature = self.auxliary_module(s1)
                outs.insert(0, aux_feature)

        return tuple(outs)
