# Copyright (c) OpenMMLab. All rights reserved.
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
            # If a layer has maximum channels, then the complementary layer
            # also has maximum channels
            subnet_dict_comp[key] = val if val.sum() == val.numel(
            ) else ~val.flip(1)
        return subnet_dict_comp
