# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_op import (DynamicBatchNorm,
                                                     build_switchable_bn)
from mmrazor.models.mutables import SlimmableChannelMutable, base_mutable
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator

NONPASS_MODULES = (nn.Conv2d, nn.Linear)
PASS_MODULES = (_BatchNorm, )


@MODELS.register_module()
class SlimmableChannelMutator(ChannelMutator):
    """Slimmable channel mutable based channel mutator.

    Args:
        channel_cfgs (list[Dict]): A list of candidate channel configs.
        mutable_cfg (dict): The config for the channel mutable.
        skip_prefixes (List[str] | Optional): The module whose name start with
            a string in skip_prefixes will not be pruned.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 channel_cfgs: List[Dict],
                 mutable_cfg: Dict,
                 skip_prefixes: Optional[List[str]] = None,
                 init_cfg: Optional[Dict] = None):
        super(SlimmableChannelMutator, self).__init__(
            mutable_cfg=mutable_cfg,
            skip_prefixes=skip_prefixes,
            init_cfg=init_cfg)

        self.channel_cfgs = self._merge_channel_cfgs(channel_cfgs)

    def _merge_channel_cfgs(self, channel_cfgs: List[Dict]):
        """Merge several channel configs.

        Args:
            channel_cfgs (List[Dict])
        """
        merged_channel_cfg = dict()
        num_subnet = len(channel_cfgs)

        for module_name in channel_cfgs[0].keys():
            channels_per_layer = [
                channel_cfgs[idx][module_name] for idx in range(num_subnet)
            ]
            merged_channels_per_layer = dict()
            for key in channels_per_layer[0].keys():
                merged_channels = [
                    channels_per_layer[idx][key] for idx in range(num_subnet)
                ]
                merged_channels_per_layer[key] = merged_channels
            merged_channel_cfg[module_name] = merged_channels_per_layer

        return merged_channel_cfg

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Note:
            Different from `ChannelMutator`, we only support Case 1 in
            `ChannelMutator`. The input supernet should be made up of original
            nn.Module. And we replace the conv/linear/bn modules in the input
            supernet with dynamic ops first. Then we convert the
            ``DynamicBatchNorm`` in supernet with ``SwitchableBatchNorm2d``.
            Finally, we set the candidate channel numbers to the corresponding
            `SlimmableChannelMutable`.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """
        self.convert_dynamic_module(supernet, self.dynamic_layer)
        print(supernet)
        self.convert_switchable_bn(supernet)
        self.set_candidate_choices(supernet)
        # The mapping from a module name to the module
        self._name2module = dict(supernet.named_modules())

    def set_candidate_choices(self, supernet):
        """Set the ``candidate_choices`` of each ``SlimmableChannelMutable``.

        Notes:
            Different from other ``OneShotChannelMutable``,
            ``candidate_choices`` is optional when instantiating a
            ``SlimmableChannelMutable``
        """
        for name, module in supernet.named_modules():
            if isinstance(module, SlimmableChannelMutable):
                candidate_choices = self.channel_cfgs[name]['current_choice']
                module.candidate_choices = candidate_choices

    def convert_switchable_bn(self, supernet):
        """Replace ``DynamicBatchNorm`` in supernet with
        ``SwitchableBatchNorm2d``.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be converted
                in your algorithm.
        """

        def traverse(module, prefix):
            for name, child in module.named_children():
                module_name = prefix + name
                if isinstance(child, DynamicBatchNorm):
                    mutable_cfg = copy.deepcopy(self.mutable_cfg)
                    key = module_name + '.mutable_num_features'
                    candidate_choices = self.channel_cfgs[key][
                        'current_choice']
                    mutable_cfg.update(
                        dict(candidate_choices=candidate_choices))
                    sbn = build_switchable_bn(child, module_name, mutable_cfg)
                    setattr(module, name, sbn)
                else:
                    traverse(child, module_name + '.')

        traverse(supernet, '')

    def switch_choices(self, idx):
        """Switch the channel config of the supernet according to input `idx`.

        If we train more than one subnet together, we need to switch the
        channel_cfg from one to another during one training iteration.

        Args:
            idx (int): The index of the current subnet.
        """
        for name, module in self.name2module.items():
            if hasattr(module, 'mutable_out'):
                module.mutable_out.current_choice = idx
            if hasattr(module, 'mutable_in'):
                module.mutable_in.current_choice = idx

    def build_search_groups(self, supernet: Module):
        """Build `search_groups`.

        The mutables in the same group should be pruned together.
        """
        pass

    def mutable_class_type(self):
        """One-shot channel mutable class type.

        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return base_mutable
