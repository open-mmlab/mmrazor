# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch.nn as nn
from mmengine import fileio

from mmrazor.structures.subnet.fix_subnet import _dynamic_to_static
from .registry import MODELS


# TO DO: Add more sub_model
# manage sub models for downstream repos
@MODELS.register_module()
def sub_model_prune(architecture,
                    mutator_cfg,
                    init_cfg: Optional[Dict] = None) -> nn.Module:
    if isinstance(mutator_cfg, str):
        mutator_cfg = fileio.load(mutator_cfg)
    mutator_cfg['parse_cfg'] = {'type': 'Config'}
    model = MODELS.build(architecture)
    mutator = MODELS.build(mutator_cfg)
    mutator.prepare_from_supernet(model)
    mutator.set_choices(mutator.current_choices)
    _dynamic_to_static(model)
    return model
