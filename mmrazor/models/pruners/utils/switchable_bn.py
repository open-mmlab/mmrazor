# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


class SwitchableBatchNorm2d(nn.Module):
    """Employs independent batch normalization for different switches in a
    slimmable network.

    To train slimmable networks, ``SwitchableBatchNorm2d`` privatizes all
    batch normalization layers for each switch in a slimmable network.
    Compared with the naive training approach, it solves the problem of feature
    aggregation inconsistency between different switches by independently
    normalizing the feature mean and variance during testing.

    Args:
        max_num_features (int): The maximum ``num_features`` among BatchNorm2d
            in all the switches.
        num_bns (int): The number of different switches in the slimmable
            networks.
    """

    def __init__(self, max_num_features, num_bns):
        super(SwitchableBatchNorm2d, self).__init__()

        self.max_num_features = max_num_features
        # number of BatchNorm2d in a SwitchableBatchNorm2d
        self.num_bns = num_bns
        bns = []
        for _ in range(num_bns):
            bns.append(nn.BatchNorm2d(max_num_features))
        self.bns = nn.ModuleList(bns)
        # When switching bn we should switch index simultaneously
        self.index = 0

    def forward(self, input):
        """Forward computation according to the current switch of the slimmable
        networks."""
        return self.bns[self.index](input)
