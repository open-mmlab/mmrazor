# Copyright (c) OpenMMLab. All rights reserved.
import json
import types
from typing import Union

import torch.nn as nn
from mmengine import fileio

from mmrazor.models.utils.expandable_utils import make_channel_divisible
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log


def post_process_for_mmdeploy_wrapper(divisor):

    def post_process_for_mmdeploy(model: nn.Module):
        s = make_channel_divisible(model, divisor=divisor)
        print_log(f'structure after make divisible: {json.dumps(s,indent=4)}')

    return post_process_for_mmdeploy


@MODELS.register_module()
def GroupFisherDeploySubModel(architecture,
                              fix_subnet: Union[dict, str] = {},
                              divisor=1,
                              parse_cfg=dict(
                                  _scope_='mmrazor',
                                  type='ChannelAnalyzer',
                                  demo_input=(1, 3, 224, 224),
                                  tracer_type='FxTracer'),
                              **kwargs):
    """Convert a architecture to a pruned static architecture for mmdeploy.

    Args:
        architecture (Union[nn.Module, dict]): the model to be pruned.
        fix_subnet (Union[dict, str]): the channel remaining ratio for each
            unit, or the path of a file including this info. Defaults to {}.
        divisor (int, optional): The divisor to make the channel number
            divisible. Defaults to 1.
        parse_cfg (dict, optional): The args for channel mutator.
    Returns:
        BaseModel: a BaseModel of mmengine.
    """
    # import avoid circular import
    from mmrazor.models.mutables import SequentialMutableChannelUnit
    from mmrazor.models.mutators import ChannelMutator
    from mmrazor.models.utils.expandable_utils.unit import ExpandableUnit

    #  build architecture
    if isinstance(architecture, dict):
        architecture = MODELS.build(architecture)
    assert isinstance(architecture, nn.Module)

    # to dynamic model
    mutator = ChannelMutator[ExpandableUnit](
        channel_unit_cfg=SequentialMutableChannelUnit, parse_cfg=parse_cfg)

    mutator.prepare_from_supernet(architecture)
    if isinstance(fix_subnet, str):
        fix_subnet = fileio.load(fix_subnet)
    assert isinstance(fix_subnet, dict)
    mutator.set_choices(fix_subnet)
    print_log(json.dumps(mutator.current_choices, indent=4))

    fix_subnet = export_fix_subnet(architecture)[0]
    load_fix_subnet(architecture, fix_subnet)

    # cooperate with mmdeploy to make the channel divisible after load
    # the checkpoint.
    if divisor != 1:
        setattr(
            architecture, 'post_process_for_mmdeploy',
            types.MethodType(
                post_process_for_mmdeploy_wrapper(divisor), architecture))
    return architecture
