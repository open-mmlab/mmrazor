# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import Tensor

from mmrazor.registry import MODELS

@MODELS.register_module()
class PruneBackbone(nn.Module):
    """Backbone of Prune Network. Reset Backbone param by subnet.yaml.

    Args:
    """

    def __init__(self,
                 source_model:  Union[BaseModel, Dict],
                 fix_subnet: Optional[ValidFixMutable] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)

        self.source_model = MODELS.build(source_model)
        self.fix_subnet = fix_subnet
        self.init_cfg = init_cfg
        self.update(self, fix_subnet)

    def update_subnet(self, fix_subnet):


    def forward(self, x):
        return self.forward(x)
