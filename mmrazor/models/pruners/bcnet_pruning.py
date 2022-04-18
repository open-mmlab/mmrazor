# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.models.builder import PRUNERS
from .ratio_pruning import RatioPruner


@PRUNERS.register_module()
class BCNetPruner(RatioPruner):

    def __init__(self, **kwargs):
        super(BCNetPruner, self).__init__(**kwargs)

    def reverse_subnet(self, subnet_dict):
        return {key: val.flip(1) for key, val in subnet_dict.items()}

    def get_complementary_subnet(self, subnet_dict):
        subnet_dict_comp = dict()
        for key, val in subnet_dict.items():
            min_channels = round(val.numel() * self.min_ratio)
            max_channels = round(val.numel() * self.max_ratio)
            channels_cur = val.sum().item()
            channels_comp = int(min_channels + max_channels - channels_cur)
            mask_comp = torch.zeros_like(val)
            mask_comp[:, :channels_comp] = 1.
            subnet_dict_comp[key] = mask_comp
        return subnet_dict_comp
