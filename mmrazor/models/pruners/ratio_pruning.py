# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from mmrazor.models.builder import PRUNERS
from .structure_pruning import StructurePruner
from .utils import SwitchableBatchNorm2d


@PRUNERS.register_module()
class RatioPruner(StructurePruner):
    """A random ratio pruner.

    Each layer can adjust its own width ratio randomly and independently.

    Args:
        ratios (list | tuple): Width ratio of each layer can be
            chosen from `ratios` randomly. The width ratio is the ratio between
            the number of reserved channels and that of all channels in a
            layer. For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer. Default to None.
    """

    def __init__(self, ratios, **kwargs):
        super(RatioPruner, self).__init__(**kwargs)
        ratios = list(ratios)
        ratios.sort()
        self.ratios = ratios
        self.min_ratio = ratios[0]

    def get_channel_mask(self, out_mask):
        """Randomly choose a width ratio of a layer from ``ratios``"""
        out_channels = out_mask.size(1)
        random_ratio = np.random.choice(self.ratios)
        new_channels = int(round(out_channels * random_ratio))
        assert new_channels > 0, \
            'Output channels should be a positive integer.'
        new_out_mask = torch.zeros_like(out_mask)
        new_out_mask[:, :new_channels] = 1

        return new_out_mask

    def sample_subnet(self):
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask(out_mask)
        return subnet_dict

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        subnet_dict = dict()
        for space_id, out_mask in self.channel_spaces.items():
            out_channels = out_mask.size(1)
            random_ratio = self.min_ratio
            new_channels = int(round(out_channels * random_ratio))
            assert new_channels > 0, \
                'Output channels should be a positive integer.'
            new_out_mask = torch.zeros_like(out_mask)
            new_out_mask[:, :new_channels] = 1

            subnet_dict[space_id] = new_out_mask

        self.set_subnet(subnet_dict)

    def switch_subnet(self, channel_cfg, subnet_ind=None):
        """Switch the channel config of the supernet according to channel_cfg.

        If we train more than one subnet together, we need to switch the
        channel_cfg from one to another during one training iteration.

        Args:
            channel_cfg (dict): The channel config of a subnet. Key is space_id
                and value is a dict which includes out_channels (and
                in_channels if exists).
            subnet_ind (int, optional): The index of the current subnet. If
                we replace normal BatchNorm2d with ``SwitchableBatchNorm2d``,
                we should switch the index of ``SwitchableBatchNorm2d`` when
                switch subnet. Defaults to None.
        """
        subnet_dict = dict()
        for name, channels_per_layer in channel_cfg.items():
            module = self.name2module[name]
            if (isinstance(module, SwitchableBatchNorm2d)
                    and subnet_ind is not None):
                # When switching bn we should switch index simultaneously
                module.index = subnet_ind
                continue

            out_channels = channels_per_layer['out_channels']
            out_mask = torch.zeros_like(module.out_mask)
            out_mask[:, :out_channels] = 1

            space_id = self.get_space_id(name)
            if space_id in subnet_dict:
                assert torch.equal(subnet_dict[space_id], out_mask)
            elif space_id is not None:
                subnet_dict[space_id] = out_mask

        self.set_subnet(subnet_dict)

    def convert_switchable_bn(self, module, num_bns):
        """Convert normal ``nn.BatchNorm2d`` to ``SwitchableBatchNorm2d``.

        Args:
            module (:obj:`torch.nn.Module`): The module to be converted.
            num_bns (int): The number of ``nn.BatchNorm2d`` in a
                ``SwitchableBatchNorm2d``.

        Return:
            :obj:`torch.nn.Module`: The converted module. Each
                ``nn.BatchNorm2d`` in this module has been converted to a
                ``SwitchableBatchNorm2d``.
        """
        module_output = module
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module_output = SwitchableBatchNorm2d(module.num_features, num_bns)

        for name, child in module.named_children():
            module_output.add_module(
                name, self.convert_switchable_bn(child, num_bns))

        del module
        return module_output
