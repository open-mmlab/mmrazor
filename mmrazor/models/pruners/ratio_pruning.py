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
        self.channel_bins = len(ratios)
        ratios = list(ratios)
        ratios.sort()
        self.ratios = ratios
        self.min_ratio = ratios[0]
        assert self.min_ratio > 0, \
            'All the numbers in ``ratios`` should be positive.'

    def sample_subnet(self):
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = dict()
        for space_id, channel_bin_mask in self.search_spaces.items():
            num_channel_bin = np.random.randint(1, self.channel_bins + 1)
            new_channel_bin_mask = torch.zeros_like(channel_bin_mask).bool()
            new_channel_bin_mask[:num_channel_bin] = True
            subnet_dict[space_id] = new_channel_bin_mask
        return subnet_dict

    def channel_bin2channel(self, channel_bin_mask, channel_mask):
        num_channels = channel_mask.size(1)
        ratio = self.ratios[channel_bin_mask.sum().item() - 1]
        out_channels = round(num_channels * ratio)
        new_channel_mask = torch.zeros_like(channel_mask).bool()
        new_channel_mask[:, :out_channels] = True

        return new_channel_mask

    def channel2channel_bin(self, channel_mask):
        num_channels = channel_mask.numel()
        out_channels = channel_mask.sum().item()
        ratio = out_channels / num_channels
        ratio_ind = self.ratios.index(
            min(self.ratios, key=lambda x: abs(x - ratio)))
        out_channel_bin = ratio_ind + 1
        channel_bin_mask = torch.zeros((self.channel_bins, )).bool()
        channel_bin_mask[:out_channel_bin] = True

        return channel_bin_mask

    def set_subnet(self, subnet_dict):
        """Modify the in_mask and out_mask of modules in supernet according to
        subnet_dict.

        Args:
            subnet_dict (dict): the key is space_id and the value is the
                corresponding sampled out_mask.
        """
        for module_name in self.modules_have_child:
            space_id = self.get_space_id(module_name)
            module = self.name2module[module_name]
            out_mask = self.channel_bin2channel(subnet_dict[space_id],
                                                module.out_mask)
            module.out_mask = out_mask.to(module.out_mask.device)

        for bn, conv in self.bn_conv_links.items():
            module = self.name2module[bn]
            conv_space_id = self.get_space_id(conv)
            # conv_space_id is None means the conv layer in front of
            # this bn module can not be pruned. So we should not set
            # the out_mask of this bn layer
            if conv_space_id is not None:
                out_mask = self.channel_bin2channel(subnet_dict[conv_space_id],
                                                    module.out_mask)
                module.out_mask = out_mask.to(module.out_mask.device)

        for module_name in self.modules_have_ancest:
            module = self.name2module[module_name]
            parents = self.node2parents[module_name]
            # To avoid ambiguity, we only allow the following two cases:
            # 1. all elements in parents are ``Conv2d``,
            # 2. there is only one element in parents, ``concat`` or ``chunk``
            # In case 1, all the ``Conv2d`` share the same space_id and
            # out_mask.
            # So in all cases, we only need the very first element in parents
            parent = parents[0]
            space_id = self.get_space_id(parent)

            if isinstance(space_id, dict):
                if 'concat' in space_id:
                    in_mask = []
                    for parent_space_id in space_id['concat']:
                        parent_out_mask = self.channel_bin2channel(
                            subnet_dict[parent_space_id],
                            self.space_id2out_mask[parent_space_id])
                        in_mask.append(parent_out_mask)
                    module.in_mask = torch.cat(
                        in_mask, dim=1).to(module.in_mask.device)
            else:
                parent_out_mask = self.channel_bin2channel(
                    subnet_dict[space_id], self.space_id2out_mask[space_id])
                module.in_mask = parent_out_mask.to(module.in_mask.device)

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        subnet_dict = dict()
        for space_id, channel_bin_mask in self.search_spaces.items():
            new_channel_bin_mask = torch.zeros_like(channel_bin_mask).bool()
            new_channel_bin_mask[0] = True
            subnet_dict[space_id] = new_channel_bin_mask

        self.set_subnet(subnet_dict)

    def set_max_channel(self):
        """Set the number of channels each layer to maximum."""
        subnet_dict = dict()
        for space_id, channel_bin_mask in self.search_spaces.items():
            subnet_dict[space_id] = torch.ones_like(channel_bin_mask).bool()
        self.set_subnet(subnet_dict)

    def get_max_channel_bins(self):
        """Get the max number of channel bins of all the groups which can be
        pruned during searching.

        Args:
            max_channel_bins (int): The max number of bins in each layer.
        """
        channel_bins_dict = dict()
        for space_id in self.search_spaces.keys():
            channel_bins_dict[space_id] = torch.ones(
                (self.channel_bins, )).bool()
        return channel_bins_dict

    def build_search_spaces(self):
        """Build channel search space.

        Args:
            name2module (dict): A mapping between module_name and module.

        Return:
            dict: The channel search space. The key is space_id and the value
                is the corresponding out_mask.
        """
        search_spaces = dict()
        self.space_id2out_mask = dict()

        for module_name in self.modules_have_child:
            need_prune = True
            for key in self.except_start_keys:
                if module_name.startswith(key):
                    need_prune = False
                    break
            if not need_prune:
                continue
            if module_name in self.module2group:
                space_id = self.module2group[module_name]
            else:
                space_id = module_name
            if space_id not in search_spaces:
                search_spaces[space_id] = torch.ones(self.channel_bins)
                module = self.name2module[module_name]
                self.space_id2out_mask[space_id] = module.out_mask

        return search_spaces

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
            channel_mask = torch.zeros_like(module.out_mask).bool()
            channel_mask[:, :out_channels] = True
            out_mask = self.channel2channel_bin(channel_mask)
            print(channels_per_layer['out_channels'],
                  channels_per_layer['raw_out_channels'],
                  out_mask.sum().item())

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
