# Copyright (c) OpenMMLab. All rights reserved.
import json
import types
from typing import Dict, Optional

import torch.nn as nn
from mmengine.model import BaseModel, BaseModule

from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log
from ...algorithms import BaseAlgorithm


def clean_params_init_info(model: nn.Module):
    """Clean param init info."""
    if hasattr(model, '_params_init_info'):
        delattr(model, '_params_init_info')
    for module in model.modules():
        if hasattr(module, '_params_init_info'):
            delattr(module, '_params_init_info')


def clean_init_cfg(model: BaseModule):
    """Clean init cfg."""
    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, BaseModule):
            module.init_cfg = {}


def empty_init_weights(model):
    """A empty init_weight method."""
    pass


def to_static_model(algorithm: BaseAlgorithm):
    """Convert a pruning algorithm to a static model."""
    if hasattr(algorithm, 'to_static'):
        model = algorithm.to_static()
    else:
        fix_subnet = export_fix_subnet(algorithm.architecture)[0]
        load_fix_subnet(algorithm.architecture, fix_subnet)
        model = algorithm.architecture

    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(empty_init_weights, model)
    return model


@MODELS.register_module()
def PruneSubModel(
    algorithm,
    divisor=1,
    mutable_cfg: Optional[Dict] = None,
    data_preprocessor=None,
):
    """A sub model for pruning algorithm.

    Args:
        algorithm (Union[BaseAlgorithm, dict]): The pruning algorithm to
            finetune.
        divisor (int): The divisor to make the channel number
            divisible. Defaults to 1.
        mutable_cfg (dict, str): the mutable choice config to change the
            model structure. Defaults to None.
        data_preprocessor (dict, optional): Placeholder for data_preprocessor.
            Defaults to None.

    Returns:
        nn.Module: a static model.
    """
    # import to avoid circular import
    from ..expandable_utils import make_channel_divisible

    # init algorithm
    if isinstance(algorithm, dict):
        algorithm = MODELS.build(algorithm)  # type: ignore
    assert isinstance(algorithm, BaseAlgorithm)
    algorithm.init_weights()
    clean_params_init_info(algorithm)

    if mutable_cfg is not None:
        assert isinstance(mutable_cfg, dict)
        algorithm.mutator.set_choices(mutable_cfg)

    print_log('PruneSubModel get pruning structure:')
    print_log(json.dumps(algorithm.mutator.choice_template, indent=4))

    # to static model
    fix_subnet = export_fix_subnet(algorithm.architecture)[0]
    load_fix_subnet(algorithm.architecture, fix_subnet)
    model = algorithm.architecture

    # make channel divisible
    if divisor != 1:
        divisible_structure = make_channel_divisible(
            model, divisor=divisor, zero_weight=False)

        print_log('PruneSubModel get divisible pruning structure:')
        print_log(json.dumps(divisible_structure, indent=4))

    # refine model
    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(empty_init_weights, model)
    return model
