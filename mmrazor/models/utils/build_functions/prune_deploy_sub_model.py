# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Union

import torch.nn as nn
from mmengine import fileio

from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log


@MODELS.register_module()
def PruneDeploySubModel(architecture,
                        mutable_cfg: Union[dict, str] = {},
                        divisor=1,
                        data_preprocessor=None,
                        init_cfg=None):
    """A submodel to deploy a pruned model.

    Args:
        architecture (Union[nn.Module, dict]): the model to be pruned.
        mutable_cfg (Union[dict, str]): the channel remaining ratio for each
            unit, or the path of a file including this info. Defaults to {}.
        divisor (int, optional): The divisor to make the channel number
            divisible. Defaults to 1.
        data_preprocessor (dict, optional): Placeholder for data_preprocessor.
            Defaults to None.
        init_cfg (dict, optional): Placeholder for init_cfg. Defaults to None.

    Returns:
        BaseModel: a BaseModel of mmengine.
    """
    # import avoid circular import
    from mmrazor.models.mutables import SequentialMutableChannelUnit
    from mmrazor.models.mutators import ChannelMutator
    from ..expandable_utils.unit import ExpandableUnit

    #  build architecture
    if isinstance(architecture, dict):
        architecture = MODELS.build(architecture)
    assert isinstance(architecture, nn.Module)

    # to dynamic model

    mutator = ChannelMutator[ExpandableUnit](
        channel_unit_cfg=SequentialMutableChannelUnit)
    mutator.prepare_from_supernet(architecture)
    if isinstance(mutable_cfg, str):
        mutable_cfg = fileio.load(mutable_cfg)
    assert isinstance(mutable_cfg, dict)
    mutator.set_choices(mutable_cfg)
    print_log(json.dumps(mutator.current_choices, indent=4))

    fix_subnet = export_fix_subnet(architecture)[0]
    load_fix_subnet(architecture, fix_subnet)

    # cooperate with mmdeploy to make the channel divisible after load
    # the checkpoint.
    if divisor != 1:
        setattr(architecture, '_razor_divisor', divisor)

    return architecture
