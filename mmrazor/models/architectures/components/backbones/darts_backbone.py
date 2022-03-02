# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcls.models.builder import BACKBONES
from mmcv.cnn import build_activation_layer, build_norm_layer

from ...utils import Placeholder


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise (stride=2)."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
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

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class StandardConv(nn.Module):
    """
    Standard conv: ReLU - Conv - BN
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
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

    def forward(self, x):
        return self.net(x)


class Node(nn.Module):

    def __init__(self, node_id, num_prev_nodes, channels,
                 num_downsample_nodes):
        super().__init__()
        edges = nn.ModuleDict()
        for i in range(num_prev_nodes):
            if i < num_downsample_nodes:
                stride = 2
            else:
                stride = 1

            edge_id = '{}_p{}'.format(node_id, i)
            edges.add_module(
                edge_id,
                nn.Sequential(
                    Placeholder(
                        group='node',
                        space_id=edge_id,
                        choice_args=dict(
                            stride=stride,
                            in_channels=channels,
                            out_channels=channels)), ))

        self.edges = Placeholder(
            group='node_edge', space_id=node_id, choices=edges)

    def forward(self, prev_nodes):
        return self.edges(prev_nodes)


class Cell(nn.Module):

    def __init__(self,
                 num_nodes,
                 channels,
                 prev_channels,
                 prev_prev_channels,
                 reduction,
                 prev_reduction,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
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
                Node(node_id, depth, channels, num_downsample_nodes))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell,
        # respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.nodes:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output


class AuxiliaryModule(nn.Module):
    """Auxiliary head in 2/3 place of network to let the gradient flow well."""

    def __init__(self,
                 in_channels,
                 base_channels,
                 out_channels,
                 norm_cfg=dict(type='BN')):

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

    def forward(self, x):
        return self.net(x)


@BACKBONES.register_module()
class DartsBackbone(nn.Module):

    def __init__(self,
                 in_channels,
                 base_channels,
                 num_layers=8,
                 num_nodes=4,
                 stem_multiplier=3,
                 out_indices=(7, ),
                 auxliary=False,
                 aux_channels=None,
                 aux_out_channels=None,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
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
        # [!] prev_prev_channels and prev_channels is output channel size,
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
            if i == self.num_layers // 3 or i == 2 * self.num_layers // 3:
                self.out_channels *= 2
                reduction = True

            cell = Cell(self.num_nodes, self.out_channels, prev_channels,
                        prev_prev_channels, reduction, prev_reduction,
                        self.act_cfg, self.norm_cfg)
            self.cells.append(cell)

            prev_prev_channels = prev_channels
            prev_channels = self.out_channels * self.num_nodes

            if i == self.auxliary_indice:
                self.auxliary_module = AuxiliaryModule(prev_channels,
                                                       self.aux_channels,
                                                       self.aux_out_channels,
                                                       self.norm_cfg)

    def forward(self, x):
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
