# Copyright (c) OpenMMLab. All rights reserved.
import json
import types

import torch.nn as nn
from mmengine import dist, fileio
from mmengine.model import BaseModel, BaseModule

from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.models.utils.expandable_utils import make_channel_divisible
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import RuntimeInfo, print_log


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


def hacky_init_weights_wrapper(fix_subnet):
    """This init weight method is used to prevent the model init again after
    build.

    Besides, It also save fix_subnet.json after RuntimeInfo is ready.
    """

    def hacky_init_weights(model):
        if dist.get_rank() == 0:
            try:
                work_dir = RuntimeInfo.work_dir()
                fileio.dump(
                    fix_subnet, work_dir + '/fix_subnet.json', indent=4)
                print_log(
                    f'save pruning structure in {work_dir}/fix_subnet.json')
            except Exception:
                pass

    return hacky_init_weights


@MODELS.register_module()
def GroupFisherSubModel(
    algorithm,
    divisor=1,
    **kargs,
):
    """Convert a algorithm(with an architecture) to a static pruned
    architecture.

    Args:
        algorithm (Union[BaseAlgorithm, dict]): The pruning algorithm to
            finetune.
        divisor (int): The divisor to make the channel number
            divisible. Defaults to 1.

    Returns:
        nn.Module: a static model.
    """
    # init algorithm
    if isinstance(algorithm, dict):
        algorithm = MODELS.build(algorithm)  # type: ignore
    assert isinstance(algorithm, BaseAlgorithm)
    algorithm.init_weights()
    clean_params_init_info(algorithm)

    pruning_structure = algorithm.mutator.choice_template
    print_log('PruneSubModel get pruning structure:')
    print_log(json.dumps(pruning_structure, indent=4))

    # to static model
    fix_mutable = export_fix_subnet(algorithm.architecture)[0]
    load_fix_subnet(algorithm.architecture, fix_mutable)
    model = algorithm.architecture

    # make channel divisible
    if divisor != 1:
        divisible_structure = make_channel_divisible(
            model, divisor=divisor, zero_weight=False)

        print_log('PruneSubModel get divisible pruning structure:')
        print_log(json.dumps(divisible_structure, indent=4))
        pruning_structure = divisible_structure

    # refine model
    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(
            hacky_init_weights_wrapper(pruning_structure), model)
    return model
